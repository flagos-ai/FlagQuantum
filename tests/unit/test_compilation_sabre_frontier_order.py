"""Preserve source placement and ready-order behavior on parallel layers."""

import pytest

import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.topology.sabre import _plan_sabre_trial
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", (4, 32, 128))
def test_parallel_local_layers_keep_their_source_order(width: int) -> None:
    instructions = tuple(
        Instruction("cz", (wire, wire + 1))
        for _ in range(24)
        for wire in range(0, width, 2)
    )
    source = CircuitIR(width, instructions)
    routed = route_to_topology(source, CouplingMap.line(width), strategy="sabre")

    assert tuple((op.name, op.wires) for op in routed) == tuple(
        (op.name, op.wires) for op in instructions
    )
    assert routed.metadata["routing"]["inserted_swap_count"] == 0
    assert routed.metadata["routing"]["mapping_restored"] is True


def test_blocked_front_precedes_new_successors_without_changing_placements() -> None:
    source = fq.Circuit(6).cx(0, 5).h(1).x(2).rz(1, theta=0.3)
    source.cz(2, 3).cx(0, 1).h(4)
    plan = _plan_sabre_trial(source.to_ir(), CouplingMap.line(6))

    assert plan is not None
    # The distant CX waits while local instructions release successors. Once
    # routed, it must still precede its dependent CX on logical qubit zero.
    assert plan.sequence == (1, 2, 6, 3, 4, 0, 5)
    assert plan.placements == ((1, 2), (1,), (2,), (1,), (2, 3), (1, 0), (4,))
    assert plan.swap_placements == (5, 5, 5, 5)
    assert plan.swaps == ((0, 1), (4, 5), (3, 4), (2, 3))
