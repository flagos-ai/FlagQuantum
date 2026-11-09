"""Symmetric difference compaction, occupied outputs and reversible cleanup."""

from collections import Counter
from itertools import combinations, product

import pytest
import torch

from flagquantum.compiler.registers.sorting.symmetric_difference import (
    symmetric_difference_ir,
    symmetric_difference_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(lhs, rhs, width, size):
    work, slots = symmetric_difference_requirements(lhs, rhs, width, size)
    words = tuple(
        tuple(range(i * width, (i + 1) * width)) for i in range(lhs + rhs + size)
    )
    flag = (lhs + rhs + size) * width
    return (
        words[:lhs],
        words[lhs : lhs + rhs],
        words[lhs + rhs :],
        flag,
        tuple(range(flag + 1, flag + 1 + work)),
        tuple(range(slots)),
    )


def encode(a, b, out, flag, width, work):
    return (
        tuple(
            (word >> i) & 1 for word in (*a, *b, *out) for i in reversed(range(width))
        )
        + (flag,)
        + (0,) * work
    )


def rows(lhs, rhs, width, size, work):
    mask = (1 << width) - 1
    occupied = tuple((i + 1) & mask for i in range(size))
    for a, b in product(
        combinations(range(1 << width), lhs), combinations(range(1 << width), rhs)
    ):
        difference = sorted(set(a) ^ set(b))
        match = len(difference) == size
        for old, flag in (((0,) * size, 0), (occupied, 1)):
            updated = (
                tuple(x ^ y for x, y in zip(old, difference, strict=True))
                if match
                else old
            )
            yield encode(a, b, old, flag, width, work), encode(
                a, b, updated, flag ^ match, width, work
            )


@pytest.mark.parametrize(
    "lhs,rhs,width,size",
    [
        (0, 0, 1, 0),
        (1, 0, 2, 1),
        (2, 0, 2, 1),
        (1, 1, 1, 0),
        (1, 1, 1, 2),
        (2, 1, 2, 2),
        (2, 1, 2, 0),
        (2, 1, 2, 4),
        (2, 2, 2, 2),
        (3, 2, 2, 1),
        (3, 2, 2, 3),
        (3, 3, 2, 4),
        (4, 2, 3, 4),
    ],
)
def test_all_set_pairs_targets_histories_and_self_inverse(lhs, rhs, width, size):
    bound = layout(lhs, rhs, width, size)
    before, after = zip(*rows(lhs, rhs, width, size, len(bound[4])), strict=True)
    gates = symmetric_difference_ir(*bound)
    counts = Counter(gate.name for gate in gates)
    assert certified_and_channel_map(gates, before) == (after, counts["measure"])
    assert certified_and_channel_map(gates, after) == (before, counts["measure"])
    assert counts["measure"] == counts["reset"]
    if lhs == rhs == 2 and width == size == 2:
        assert len(bound[4]) == 37
        assert counts["measure"] == 97
        assert counts["t"] + counts["tdg"] == 752


@pytest.mark.parametrize(
    "lhs,rhs,width,size", [(1, 1, 1, 0), (1, 1, 1, 2), (2, 2, 2, 2)]
)
def test_append_complex_superposition_and_work_cleanup(lhs, rhs, width, size):
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    bound = layout(lhs, rhs, width, size)
    entries = tuple(rows(lhs, rhs, width, size, len(bound[4])))
    qubits = len(entries[0][0]) + 1
    gates = symmetric_difference_ir(*bound)
    ir = DynamicCircuit(qubits).append(FieldOperation(qubits, gates)).to_ir()
    vec = torch.arange(1, 2 * len(entries) + 1, dtype=torch.float64)
    amp = torch.complex(vec, 2 - 3 * vec)
    amp /= amp.norm()
    initial, expected = {}, {}
    for i, ((before, after), spectator) in enumerate(product(entries, (0, 1))):
        for state, value in ((initial, before), (expected, after)):
            state[int("".join(map(str, (*value, spectator))), 2)] = complex(amp[i])
    if lhs == 1:
        data = torch.zeros(1 << qubits, dtype=torch.complex128)
        target = torch.zeros_like(data)
        for vector, amplitudes in ((data, initial), (target, expected)):
            for index, value in amplitudes.items():
                vector[index] = value
        events = sum(gate.name == "measure" for gate in gates)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(data, tuple(ir.instructions), history),
                target / 2 ** (events / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        for seed in (0, 9, 13):
            result = execute_sparse_register(ir, initial, seed=seed)
            assert all(
                abs(result.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in result.keys() | expected.keys()
            )


@pytest.mark.ftqc_reference
def test_locked_model_and_native_network_gate_dependencies():
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic.comparison import Equals, EqualsAConstant
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.bloqs.arithmetic.lists.symmetric_difference import SymmetricDifference
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, BitonicSort
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.resource_counting import SympySymbolAllocator

    from flagquantum.compiler.registers.comparison.comparison_layout import (
        comparison_network_requirements,
    )
    from flagquantum.core.ir import Instruction
    from tests.unit.test_bitonic_merge import reference_network
    from tests.unit.test_hamming_weight_phasing import source_count_leaf
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    model = SymmetricDifference(2, 2, 2, QUInt(2))
    with pytest.raises(DecomposeNotImplementedError):
        model.decompose_bloq()
    counts = model.build_call_graph(SympySymbolAllocator())
    assert counts[BitonicMerge(2, 2)] == counts[BitonicMerge(2, 2).adjoint()] == 1
    equal = Equals(QUInt(2))
    assert counts[equal] == counts[equal.adjoint()] == 3
    assert (
        counts[HammingWeightCompute(3)]
        == counts[HammingWeightCompute(3).adjoint()]
        == 1
    )
    constant = EqualsAConstant(2, 1)
    assert counts[constant] == counts[constant.adjoint()] == 1
    # The source uses floor division even when the desired parity is impossible.
    odd = SymmetricDifference(2, 1, 2, QUInt(2))
    assert odd.build_call_graph(SympySymbolAllocator())[EqualsAConstant(2, 0)] == 1
    assert symmetric_difference_ir(*layout(2, 1, 2, 2)) == ()

    left, right, output, flag, work, slots = layout(2, 2, 2, 2)
    gates = symmetric_difference_ir(left, right, output, flag, work, slots)
    keys = tuple(work[i : i + 3] for i in range(0, 12, 3))
    payload = tuple(key[1:] for key in keys)
    merge_history, compact_history = work[12:16], work[16:22]
    eq, pool = work[22:25], work[29:]

    def network(sorter, words, history, merge, adjoint):
        q, c = comparison_network_requirements(len(words[0]), len(history))
        return reference_network(
            sorter, words, history, pool[:q], slots[:c], merge=merge, adjoint=adjoint
        )

    forward = network(BitonicMerge(2, 2), payload, merge_history, True, False)
    backward = network(BitonicMerge(2, 2), payload, merge_history, True, True)
    assert_source(gates[8 : 8 + len(forward)], forward)
    assert_source(gates[-8 - len(backward) : -8], backward)
    cursor = 8 + len(forward)
    for a, b, target in zip(payload[:-1], payload[1:], eq, strict=True):
        compare = reference_arithmetic_gates(
            Equals(QUInt(2)),
            {"x": a, "y": b, "target": (target,)},
            pool[:1],
            {pool[0]: slots[0]},
        )
        assert_source(gates[cursor : cursor + len(compare)], compare)
        cursor += len(compare)

    cursor += 6  # Two duplicate-tag CX per adjacent flag.
    parts = {"x": eq, "junk": work[25:26], "out": work[26:28]}
    counted = source_count_leaf(HammingWeightCompute(3), {"x": eq}, parts, slots[:1])
    assert counted is not None
    count_stream, _ = counted
    assert_source(gates[cursor : cursor + len(count_stream)], count_stream)
    cursor += len(count_stream)
    select = reference_arithmetic_gates(
        constant,
        {"x": parts["out"], "target": work[28:29]},
        pool[:1],
        {pool[0]: slots[0]},
    )
    assert_source(gates[cursor : cursor + len(select)], select)
    cursor += len(select)
    compact = network(BitonicSort(4, 3), keys, compact_history, False, False)
    assert_source(gates[cursor : cursor + len(compact)], compact)
    cursor += len(compact)
    for entry, target_word in zip(payload[:2], output, strict=True):
        for x, y in zip(entry, target_word, strict=True):
            binding = {"ctrl": (work[28], x), "target": pool[:1]}
            make = reference_arithmetic_gates(And(), binding, (), {}, cirq_style=True)
            erase = reference_arithmetic_gates(
                And().adjoint(), binding, (), {pool[0]: slots[0]}, cirq_style=True
            )
            copy = (*make, Instruction("cx", (pool[0], y)), *erase)
            assert_source(gates[cursor : cursor + len(copy)], copy)
            cursor += len(copy)
    assert gates[cursor] == Instruction("cx", (work[28], flag))
    cursor += 1
    reverse = network(BitonicSort(4, 3), keys, compact_history, False, True)
    assert_source(gates[cursor : cursor + len(reverse)], reverse)
    cursor += len(reverse) + len(select)
    uncounted = source_count_leaf(
        HammingWeightCompute(3).adjoint(), parts, parts, slots[:1]
    )
    assert uncounted is not None
    uncount_stream, _ = uncounted
    assert_source(gates[cursor : cursor + len(uncount_stream)], uncount_stream)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"left": ()}, "left >= right"),
        ({"right": ((2,), (4,))}, "matching word widths"),
        ({"workspace": ()}, "requires clean work"),
        ({"flag": 0}, "distinct nonnegative wires"),
    ],
)
def test_bad_binding(change, match):
    names = ("left", "right", "output", "flag", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 2, 2, 2), strict=True))
    with pytest.raises(ValueError, match=match):
        symmetric_difference_ir(**(bound | change))


def test_word_domain_validation():
    with pytest.raises(ValueError, match="exceeds the word domain"):
        symmetric_difference_requirements(3, 2, 1, 1)
