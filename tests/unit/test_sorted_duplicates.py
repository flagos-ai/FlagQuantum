"""Source bindings, defect witnesses, and clean sorted-list flag channels."""

from collections import Counter
from itertools import combinations_with_replacement, product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.sorting.sorted_duplicates import (
    sorted_duplicate_requirements,
    sorted_duplicates_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    length: int, n: int, controlled: bool
) -> tuple[
    tuple[tuple[int, ...], ...], int, tuple[int, ...], tuple[int, ...], int | None, int
]:
    control = length * n + 1 if controlled else None
    start = length * n + 1 + controlled
    q, c = sorted_duplicate_requirements(length, n, controlled=controlled)
    words = tuple(tuple(range(i * n, (i + 1) * n)) for i in range(length))
    return (
        words,
        length * n,
        tuple(range(start, start + q)),
        tuple(range(40, 40 + c)),
        control,
        start + q,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "length,n,controlled",
    [(1, 2, True), (2, 1, False), (2, 2, True), (3, 2, False), (4, 3, True)],
)
@pytest.mark.parametrize("signed", [False, True])
def test_original_graph_gates_aliases_and_cost(
    length: int, n: int, controlled: bool, signed: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic import LinearDepthHalfLessThan
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    words, flag, work, bits, control, total = layout(length, n, controlled)
    gates, output = sorted_duplicates_ir(
        words, flag, work, bits, signed=signed, control=control, source_aliases=True
    )
    source = HasDuplicates(length, QInt(n) if signed else QUInt(n), controlled)
    src_ports = {"xs": tuple(wire for word in words for wire in word), "flag": (flag,)}
    if control is not None:
        src_ports["ctrl"] = (control,)
    history = iter(work[i : i + n + 2] for i in range(0, (length - 1) * (n + 2), n + 2))
    scratch = work[(length - 1) * (n + 2) :]

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, LinearDepthHalfLessThan):
            if bloq.uncompute:
                difference = inputs["c"]
                output = {"a": inputs["b"], "b": inputs["a"]}
                allocation = scratch[:2]
                measured = dict(zip(difference[::-1][1:], bits[:n], strict=True))
            else:
                record = next(history)
                difference, target = record[:-1], record[-1:]
                output = {**inputs, "c": difference, "target": target}
                allocation = (*scratch[:2], *difference[::-1], *target)
                measured = {}
            assert all(output[key] == wires for key, wires in desired.items())
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    allocation,
                    measured,
                    expected_outputs=output,
                    reuse_allocations=False,
                ),
                output,
            )
        if isinstance(bloq, MultiControlX):
            q = len(inputs["controls"]) - 1
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    scratch[:q],
                    dict(zip(scratch[:q], bits[:q], strict=True)),
                ),
                inputs,
            )
        if isinstance(bloq, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        return None

    expected, out_ports = walk_source_graph(source, src_ports, {}, leaf)
    assert_source(gates, expected)
    assert next(history, None) is None
    assert out_ports == {
        **src_ports,
        "xs": tuple(wire for word in output for wire in word),
    }
    clean, _ = sorted_duplicates_ir(
        words, flag, work, bits, signed=signed, control=control
    )
    counts = Counter(g.name for g in gates)
    assert Counter(g.name for g in clean) == counts
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement
    assert get_cost_value(source, QubitCount()) == total - int(1 < length < 4)


@pytest.mark.parametrize("length,n", [(1, 1), (2, 1), (3, 1), (4, 2)])
@pytest.mark.parametrize("controlled,signed", list(product((False, True), repeat=2)))
def test_clean_channel_all_sorted_inputs_and_measurements(
    length: int, n: int, controlled: bool, signed: bool
) -> None:
    words, flag, work, bits, control, total = layout(length, n, controlled)
    gates, output = sorted_duplicates_ir(
        words, flag, work, bits, signed=signed, control=control
    )
    assert output == words
    lower = -(1 << (n - 1)) if signed else 0
    domain = range(lower, lower + (1 << n))
    inputs, expected = [], []
    for values in combinations_with_replacement(domain, length):
        duplicate = len(set(values)) < length
        for target, cv in product((0, 1), range(2) if controlled else (1,)):
            row = [0] * (total + 1)
            for wires, value in zip(words, values, strict=True):
                for i, wire in enumerate(reversed(wires)):
                    row[wire] = value >> i & 1
            if control is not None:
                row[control] = cv
            row[flag] = target
            for spectator in (0, 1):
                row[-1] = spectator
                inputs.append(tuple(row))
                mapped = row.copy()
                mapped[flag] ^= duplicate and cv
                expected.append(tuple(mapped))
    events = sum(g.name == "measure" for g in gates)
    assert certified_and_channel_map(gates, tuple(inputs)) == (tuple(expected), events)
    assert certified_and_channel_map(gates, tuple(expected)) == (tuple(inputs), events)

    def address(row: tuple[int, ...]) -> int:
        return sum(bit << (total - i) for i, bit in enumerate(row))

    weights = torch.arange(1, len(inputs) + 1, dtype=torch.float64)
    amps = torch.complex(weights, weights * (weights % 5 - 2))
    amps /= amps.norm()
    before = dict(zip(map(address, inputs), amps.tolist(), strict=True))
    after = dict(zip(map(address, expected), amps.tolist(), strict=True))
    if length <= 3:
        start = torch.zeros(1 << (total + 1), dtype=torch.complex128)
        end = torch.zeros_like(start)
        start[list(before)] = amps
        end[list(after)] = amps
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(start, gates, history),
                end * 2 ** (-events / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        for seed in (3, 41):
            actual = execute_sparse_register(
                CircuitIR(total + 1, gates), before, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - after.get(i, 0)) < 2e-12
                for i in actual.keys() | after.keys()
            )


@pytest.mark.ftqc_reference
def test_original_source_failure_and_native_channel_witnesses() -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import BloqError, QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    with pytest.raises(BloqError, match="controls"):
        HasDuplicates(1, QUInt(2)).decompose_bloq()
    two = HasDuplicates(2, QUInt(2)).decompose_bloq()
    values = dict(xs=np.array([0, 1]), flag=0)
    assert tuple(two.on_classical_vals(**values)["xs"]) == (0, 1)
    assert tuple(two.flatten().on_classical_vals(**values)["xs"]) == (1, 0)
    words, flag, work, bits, ctrl, total = layout(2, 2, False)
    gates, returned = sorted_duplicates_ir(words, flag, work, bits, source_aliases=True)
    assert returned == words[::-1]
    initial = (0, 0, 0, 1, *(0,) * (total - 4))
    assert certified_and_channel_map(gates, (initial,)) == ((initial,), 2)

    three = HasDuplicates(3, QUInt(2)).decompose_bloq()
    values = dict(xs=np.array([0, 0, 1]), flag=0)
    assert three.on_classical_vals(**values)["flag"] == 1
    with pytest.raises(ValueError, match="Inconsistent.*And"):
        three.flatten().on_classical_vals(**values)
    words, flag, work, bits, ctrl, total = layout(3, 2, False)
    gates, _ = sorted_duplicates_ir(words, flag, work, bits, source_aliases=True)
    initial = (0, 0, 0, 0, 0, 1, *(0,) * (total - 6))
    with pytest.raises(AssertionError):
        certified_and_channel_map(gates, (initial,))


def test_duplicate_layout_validation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        sorted_duplicate_requirements(0, 2)
    with pytest.raises(ValueError, match="boolean"):
        sorted_duplicate_requirements(2, 1, controlled=1)
    with pytest.raises(ValueError, match="equal-width"):
        sorted_duplicates_ir(((0,), (1, 2)), 3, (), ())
    with pytest.raises(ValueError, match="boolean"):
        sorted_duplicates_ir(((0,),), 1, (), (), signed=1)
    with pytest.raises(ValueError, match="work wires"):
        sorted_duplicates_ir(((0,), (1,)), 2, (), ())
    with pytest.raises(ValueError, match="distinct"):
        sorted_duplicates_ir(((0, 1), (2, 3)), 4, tuple(range(5, 11)), (0, 0))
    with pytest.raises(ValueError, match="singleton"):
        sorted_duplicates_ir(((0,),), 1, (), (), source_aliases=True)


def test_append_sorted_duplicates_example() -> None:
    from examples.algorithms.register_sorted_duplicates import main

    main()
