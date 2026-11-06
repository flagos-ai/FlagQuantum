"""Clean in-place sort and restoration on the distinct-word subspace."""

from collections import Counter
from itertools import combinations, permutations, product

import pytest
import torch

from flagquantum.compiler.registers.in_place_sort import (
    in_place_sort_ir,
    in_place_sort_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(count, width):
    rank_width, work, slots = in_place_sort_requirements(count, width)
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(count))
    first = count * width
    ranks = tuple(
        tuple(range(first + i * rank_width, first + (i + 1) * rank_width))
        for i in range(count)
    )
    first += count * rank_width
    return words, ranks, tuple(range(first, first + work)), tuple(range(slots))


def pack(words, ranks, width, rank_width, work):
    return (
        tuple(
            (word >> j) & 1
            for values, bits in ((words, width), (ranks, rank_width))
            for word in values
            for j in reversed(range(bits))
        )
        + (0,) * work
    )


def samples(count, width, work):
    r = (count - 1).bit_length()
    for table in combinations(range(1 << width), count):
        for perm in permutations(range(count)):
            words = tuple(table[i] for i in perm)
            yield (
                pack(words, (0,) * count, width, r, work),
                pack(table, perm, width, r, work),
            )


def expected_budget(count, width):
    if count == 1:
        return 0, 0
    r = (count - 1).bit_length()
    padded = 1 << r
    key = width + (padded != count)
    comps = (padded // 2) * r * (r + 1) // 2
    rank_meas = 2 * count * (count - 1) * (width - 1)
    reconstruction_meas = count**2 * (width + 2 * (r - 1))
    index_meas = rank_meas + reconstruction_meas
    return (
        2 * comps * (15 * key - 4) + 4 * index_meas,
        2 * comps * (2 * key - 1) + index_meas,
    )


@pytest.mark.parametrize(
    "count,width", [(1, 1), (2, 1), (2, 3), (3, 2), (4, 2), (5, 3)]
)
def test_distinct_lists_inverse_all_histories_and_resource(count, width):
    bound = layout(count, width)
    before, after = zip(*samples(count, width, len(bound[2])), strict=True)
    t_count, measurement_count = expected_budget(count, width)
    for adjoint, inputs, expected in ((False, before, after), (True, after, before)):
        gates = in_place_sort_ir(*bound, adjoint=adjoint)
        assert certified_and_channel_map(gates, inputs) == (expected, measurement_count)
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == t_count
        assert counts["measure"] == counts["reset"] == measurement_count


@pytest.mark.parametrize("count,width", [(2, 1), (3, 2)])
def test_append_complex_amplitudes_and_clean_inverse(count, width):
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    bound = layout(count, width)
    rows = tuple(samples(count, width, len(bound[2])))
    size = len(rows[0][0]) + 1
    inputs, outputs = {}, {}
    values = torch.arange(1, len(rows) * 2 + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, 3 - values)
    amplitudes /= amplitudes.norm()
    for i, ((a, b), spectator) in enumerate(product(rows, (0, 1))):
        for state, raw in ((inputs, a), (outputs, b)):
            address = int("".join(map(str, (*raw, spectator))), 2)
            state[address] = complex(amplitudes[i])
    for adjoint, before, expected in (
        (False, inputs, outputs),
        (True, outputs, inputs),
    ):
        gates = in_place_sort_ir(*bound, adjoint=adjoint)
        ir = DynamicCircuit(size).append(FieldOperation(size, gates)).to_ir()
        if count == 2:
            initial_state = torch.zeros(1 << size, dtype=torch.complex128)
            expected_state = torch.zeros_like(initial_state)
            for target, source in ((initial_state, before), (expected_state, expected)):
                for index, value in source.items():
                    target[index] = value
            events = expected_budget(count, width)[1]
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(initial_state, tuple(ir.instructions), history),
                    expected_state / 2 ** (events / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            result = execute_sparse_register(ir, before, seed=81)
            assert all(
                abs(result.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in result.keys() | expected.keys()
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count,width", [(2, 2), (3, 2)])
def test_assembled_gate_stream_and_locked_sort_model(count, width):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic import Xor
    from qualtran.bloqs.arithmetic.lists.sort_in_place import SortInPlace
    from qualtran.bloqs.arithmetic.sorting import BitonicSort, Comparator
    from qualtran.resource_counting import SympySymbolAllocator

    from flagquantum.compiler.registers.comparison_layout import (
        comparison_network_requirements,
    )
    from flagquantum.compiler.registers.permutation_index import (
        permutation_index_requirements,
        permutation_rank_xor_ir,
        permutation_words_xor_ir,
    )
    from flagquantum.core.ir import Instruction
    from tests.unit.test_bitonic_merge import reference_network

    source = SortInPlace(count, QUInt(width))
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    cmp = Comparator(width)
    assert source.build_call_graph(SympySymbolAllocator()) == dict.fromkeys(
        (cmp, cmp.adjoint(), Xor(QUInt(width))), 3 * count**2
    )

    words, ranks, work, slots = layout(count, width)
    r = len(ranks[0])
    padded = 1 << r
    key_width = width + (count != padded)
    sorter = BitonicSort(padded, key_width)
    head = iter(work)
    table = tuple(tuple(next(head) for _ in range(width)) for _ in range(count))
    keys = tuple(tuple(next(head) for _ in range(key_width)) for _ in range(padded))
    history = tuple(next(head) for _ in range(sorter.num_comparisons))
    pool = tuple(head)
    q, c = comparison_network_requirements(key_width, len(history))

    def cx_words(src, dst):
        return tuple(
            Instruction("cx", (x, y))
            for a, b in zip(src, dst, strict=True)
            for x, y in zip(a, b, strict=True)
        )

    payload = tuple(key[-width:] for key in keys[:count])
    load = (
        *cx_words(words, payload),
        *(Instruction("x", (key[0],)) for key in keys[count:]),
    )
    forward = reference_network(
        sorter, keys, history, pool[:q], slots[:c], merge=False, adjoint=False
    )
    backward = reference_network(
        sorter, keys, history, pool[:q], slots[:c], merge=False, adjoint=True
    )
    copy = (*load, *forward, *cx_words(payload, table), *backward, *load)
    _, q, c = permutation_index_requirements(count, width)
    rank = permutation_rank_xor_ir(table, words, ranks, pool[:q], slots[:c])
    reconstruct = permutation_words_xor_ir(table, ranks, words, pool[:q], slots[:c])
    to_words, to_table = cx_words(table, words), cx_words(words, table)
    assert_source(
        in_place_sort_ir(words, ranks, work, slots),
        (*copy, *rank, *reconstruct, *to_words, *to_table),
    )
    assert_source(
        in_place_sort_ir(words, ranks, work, slots, adjoint=True),
        (*to_table, *to_words, *reconstruct, *rank, *copy),
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"words": ()}, "positive count"),
        ({"adjoint": 1}, "must be boolean"),
        ({"ranks": ()}, "matching word and rank"),
        ({"workspace": ()}, "requires clean work"),
        ({"ranks": ((0,), (3,))}, "distinct nonnegative wires"),
    ],
)
def test_layout_rejects_bad_bindings(change, match):
    bound = dict(
        zip(
            ("words", "ranks", "workspace", "classical_bits"), layout(2, 1), strict=True
        )
    )
    with pytest.raises(ValueError, match=match):
        in_place_sort_ir(**(bound | change))
