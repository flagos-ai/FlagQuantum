"""Permutation indices, reconstructed words and measured branch conformance."""

from collections import Counter
from itertools import combinations, permutations, product

import pytest
import torch

from flagquantum.compiler.registers.permutation_index import (
    permutation_index_requirements,
    permutation_rank_xor_ir,
    permutation_words_xor_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(count, width):
    rank_width, work, slots = permutation_index_requirements(count, width)
    table = tuple(tuple(range(i, i + width)) for i in range(0, count * width, width))
    words = tuple(tuple(q + count * width for q in word) for word in table)
    start = 2 * count * width
    ranks = tuple(
        tuple(range(start + i * rank_width, start + (i + 1) * rank_width))
        for i in range(count)
    )
    start += count * rank_width
    return table, words, ranks, tuple(range(start, start + work)), tuple(range(slots))


def packed(table, words, ranks, width, rank_width, work):
    return (
        tuple(
            (word >> bit) & 1
            for values, size in ((table, width), (words, width), (ranks, rank_width))
            for word in values
            for bit in reversed(range(size))
        )
        + (0,) * work
    )


@pytest.mark.parametrize("count,width", [(1, 2), (2, 1), (2, 3), (3, 2), (4, 3)])
def test_every_permutation_and_measurement_history(count, width):
    bound = layout(count, width)
    table, words, ranks, work, bits = bound
    rank_width = len(ranks[0])
    count_masks = (1 << rank_width) - 1
    before, ranked, erased, occupied_before, occupied_after = [], [], [], [], []
    for entries in combinations(range(1 << width), count):
        for perm in permutations(range(count)):
            original = tuple(entries[i] for i in perm)
            before.append(
                packed(entries, original, (0,) * count, width, rank_width, len(work))
            )
            ranked.append(packed(entries, original, perm, width, rank_width, len(work)))
            erased.append(
                packed(entries, (0,) * count, perm, width, rank_width, len(work))
            )
            occupied_before.append(
                packed(
                    entries,
                    original,
                    (count_masks,) * count,
                    width,
                    rank_width,
                    len(work),
                )
            )
            occupied_after.append(
                packed(
                    entries,
                    original,
                    tuple(i ^ count_masks for i in perm),
                    width,
                    rank_width,
                    len(work),
                )
            )
    rank_ir = permutation_rank_xor_ir(*bound)
    word_ir = permutation_words_xor_ir(table, ranks, words, work, bits)
    rank_measure = 2 * count * (count - 1) * (width - 1)
    word_measure = count * count * (width + 2 * (rank_width - 1)) if count > 1 else 0
    for gates, measure, initial, expected in (
        (rank_ir, rank_measure, before, ranked),
        (rank_ir, rank_measure, occupied_before, occupied_after),
        (word_ir, word_measure, ranked, erased),
        (word_ir, word_measure, erased, ranked),
    ):
        assert certified_and_channel_map(gates, tuple(initial)) == (
            tuple(expected),
            measure,
        )
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == 4 * measure
        assert counts["measure"] == counts["reset"] == measure


def test_unused_rank_pattern_preserves_occupied_word():
    table, words, ranks, work, bits = layout(3, 2)
    before = packed((0, 1, 3), (1, 2, 2), (3, 0, 2), 2, 2, len(work))
    after = packed((0, 1, 3), (1, 2, 1), (3, 0, 2), 2, 2, len(work))
    gates = permutation_words_xor_ir(table, ranks, words, work, bits)
    assert certified_and_channel_map(gates, (before,)) == ((after,), 36)


def test_append_full_complex_permutation_and_erase_every_raw_history():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    table, words, ranks, work, bits = layout(2, 1)
    gates = (
        *permutation_rank_xor_ir(table, words, ranks, work, bits),
        *permutation_words_xor_ir(table, ranks, words, work, bits),
    )
    size = 9
    ir = DynamicCircuit(size).append(FieldOperation(size, gates)).to_ir()
    initial = torch.zeros(1 << size, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    coeff = torch.tensor((1 + 2j, 3 - 1j, -2 + 4j, -3j), dtype=torch.complex128)
    coeff /= coeff.norm()
    for i, (perm, spectator) in enumerate(product(((0, 1), (1, 0)), (0, 1))):
        for state, fields in ((initial, (perm, (0, 0))), (expected, ((0, 0), perm))):
            raw = packed((0, 1), *fields, 1, 1, len(work))
            index = int("".join(map(str, (*raw, spectator))), 2)
            state[index] = coeff[i]
    for history in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(initial, tuple(ir.instructions), history),
            expected / 4,
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count,width", [(2, 2), (3, 3)])
def test_locked_signature_model_and_rank_dependency_stream(count, width):
    pytest.importorskip("qualtran")
    from qualtran import BQUInt, DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic import Xor
    from qualtran.bloqs.arithmetic.comparison import Equals, EqualsAConstant
    from qualtran.bloqs.arithmetic.lists.sort_in_place import SortInPlace
    from qualtran.bloqs.arithmetic.sorting import Comparator
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.resource_counting import SympySymbolAllocator

    dtype = QUInt(width)
    source = SortInPlace(count, dtype)
    rank_width = (count - 1).bit_length()
    assert source.signature.get_left("pi").dtype == BQUInt(rank_width, count)
    assert source.signature.get_right("pi").shape == (count,)
    model = source.build_call_graph(SympySymbolAllocator())
    assert model == {
        Comparator(width): 3 * count**2,
        Comparator(width).adjoint(): 3 * count**2,
        Xor(dtype): 3 * count**2,
    }
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()

    table, words, ranks, work, bits = layout(count, width)
    flag, *scratch = work
    scratch = tuple(scratch)
    slots = dict(zip(scratch, bits, strict=True))
    forward = []
    reconstruction = []
    for word, rank in zip(words, ranks, strict=True):
        for i, entry in enumerate(table):
            if i:
                check = reference_arithmetic_gates(
                    Equals(dtype),
                    {"x": entry, "y": word, "target": (flag,)},
                    scratch,
                    slots,
                )
                forward.extend(check)
                forward.extend(
                    Instruction("cx", (flag, target))
                    for j, target in enumerate(rank)
                    if i & (1 << (rank_width - 1 - j))
                )
                forward.extend(check)
            check = reference_arithmetic_gates(
                EqualsAConstant(rank_width, i),
                {"x": rank, "target": (flag,)},
                scratch[: rank_width - 1],
                slots,
            )
            reconstruction.extend(check)
            for x, y in zip(entry, word, strict=True):
                bindings = {"ctrl": (flag, x), "target": scratch[:1]}
                reconstruction.extend(
                    reference_arithmetic_gates(And(), bindings, (), {}, cirq_style=True)
                )
                reconstruction.append(Instruction("cx", (scratch[0], y)))
                reconstruction.extend(
                    reference_arithmetic_gates(
                        And().adjoint(), bindings, (), slots, cirq_style=True
                    )
                )
            reconstruction.extend(check)
    assert_source(
        permutation_rank_xor_ir(table, words, ranks, work, bits), tuple(forward)
    )
    assert_source(
        permutation_words_xor_ir(table, ranks, words, work, bits), tuple(reconstruction)
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"table": ()}, "positive count"),
        ({"ranks": ((6,),)}, "matching table"),
        ({"words": ((0, 5), (6, 7))}, "distinct nonnegative wires"),
        ({"workspace": ()}, "work qubits"),
        ({"classical_bits": (-1,)}, "distinct nonnegative slots"),
    ],
)
def test_layout_validation(change, match):
    bound = dict(
        zip(
            ("table", "words", "ranks", "workspace", "classical_bits"),
            layout(2, 2),
            strict=True,
        )
    )
    with pytest.raises(ValueError, match=match):
        permutation_rank_xor_ir(**(bound | change))


def test_table_cannot_exceed_word_domain():
    with pytest.raises(ValueError, match="distinct word domain"):
        permutation_index_requirements(3, 1)
