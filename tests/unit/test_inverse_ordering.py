"""Original ordering flag component graph, complete channels and peak work reuse."""

from collections import Counter
from itertools import product
from typing import Any

import pytest

from flagquantum.compiler.registers.inverse_ordering import inverse_ordering_ir
from flagquantum.core.ir import Instruction
from tests.unit.inverse_flag_checks import check_flag_channel
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    n: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        2 * n,
        2 * n + 1,
        2 * n + 2,
        2 * n + 3,
        tuple(range(2 * n + 4, 3 * n + 8)),
        tuple(range(40, 40 + max(n, 2))),
    )


def original_ordering_gates(n: int, adjoint: bool) -> tuple[Instruction, ...]:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep3

    u, v, b, a, m, f, work, bits = layout(n)
    signs, difference, greater = work[:2], work[2:-1], work[-1:]
    slots = dict(zip(difference[::-1][1:], bits[:n], strict=True))
    slots.update(zip(signs, bits[:2], strict=True))

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        underlying = getattr(bloq, "subbloq", bloq)
        name = type(underlying).__name__
        if name == "CNOT":
            assert set(inputs) == {"ctrl", "target"}
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if name == "LinearDepthHalfGreaterThan":
            outputs = {"a": inputs["a"], "b": inputs["b"]}
            erase = "c" in inputs
            if not erase:
                outputs.update(c=difference, target=greater)
            allocations = signs if erase else (*signs, *difference[::-1], *greater)
        elif name == "MultiAnd":
            outputs = {"ctrl": inputs["ctrl"]}
            erase = "target" in inputs
            if not erase:
                outputs.update(junk=signs[:1], target=signs[-1:])
            allocations = () if erase else signs
        else:
            return None
        assert all(outputs[key] == value for key, value in desired.items())
        gates = reference_arithmetic_gates(
            bloq,
            inputs,
            allocations,
            slots,
            expected_outputs=outputs,
            reuse_allocations=False,
        )
        return gates, outputs

    source = _KaliskiIterationStep3(n)
    if adjoint:
        source = source.adjoint()
    inputs = dict(u=u, v=v, b=(b,), a=(a,), m=(m,), f=(f,))
    gates, outputs = walk_source_graph(source, inputs, inputs, leaf)
    assert outputs == inputs
    return gates


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_source_graph_gates_and_independent_costs(
    n: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep3
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    u, v, b, a, m, f, work, bits = layout(n)
    gates = inverse_ordering_ir(u, v, b, a, m, f, work, bits, adjoint=adjoint)
    assert_source(gates, original_ordering_gates(n, adjoint))
    source = _KaliskiIterationStep3(n)
    if adjoint:
        source = source.adjoint()
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n + 2)
    assert counts["measure"] == counts["reset"] == model.measurement == n + 2
    assert get_cost_value(source, QubitCount()) == 3 * n + 8


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_column_all_histories_and_original_inverse(n: int) -> None:
    u, v, b, a, m, f, work, bits = layout(n)
    outputs = []
    for raw in range(1 << (2 * n + 4)):
        x, y = raw >> (n + 4), raw >> 4 & ((1 << n) - 1)
        branch, active = raw >> 3 & 1, raw & 1
        condition = int(x > y) & (branch ^ 1) & active
        outputs.append(raw ^ (condition << 2) ^ (condition << 1))
    check_flag_channel(
        inverse_ordering_ir(u, v, b, a, m, f, work, bits),
        inverse_ordering_ir(u, v, b, a, m, f, work, bits, adjoint=True),
        tuple(outputs),
        2 * n + 4,
        n + 4,
        n + 2,
        raw_branches=n <= 2,
    )


@pytest.mark.parametrize("bad", ["width", "adjoint", "alias", "work", "slots", "bits"])
def test_layout_rejections(bad: str) -> None:
    u, v, b, a, m, f, work, bits = layout(2)
    inverse = False
    if bad == "width":
        u = ()
    elif bad == "adjoint":
        inverse = 1
    elif bad == "alias":
        m = a
    elif bad == "work":
        work = work[:-1]
    elif bad == "slots":
        bits = (40,)
    else:
        bits = (40, 40)
    with pytest.raises(ValueError):
        inverse_ordering_ir(u, v, b, a, m, f, work, bits, adjoint=inverse)


def test_eight_bit_boundaries_every_flag_and_measurement_history() -> None:
    u, v, b, a, m, f, work, bits = layout(8)
    values = (0, 1, 127, 128, 254, 255)
    width, q = 32, len(work)
    before, after = [], []
    for x, y, flags in product(values, values, range(16)):
        raw = (x << 12) | (y << 4) | flags
        condition = int(x > y) & (1 ^ (flags >> 3)) & (flags & 1)
        out = raw ^ (condition << 2) ^ (condition << 1)
        before.append(tuple((raw << q) >> (width - 1 - i) & 1 for i in range(width)))
        after.append(tuple((out << q) >> (width - 1 - i) & 1 for i in range(width)))
    gates = inverse_ordering_ir(u, v, b, a, m, f, work, bits)
    result, events = certified_and_channel_map(gates, tuple(before))
    assert result == tuple(after) and events == 10
    restored, _ = certified_and_channel_map(
        inverse_ordering_ir(u, v, b, a, m, f, work, bits, adjoint=True), result
    )
    assert restored == tuple(before)
