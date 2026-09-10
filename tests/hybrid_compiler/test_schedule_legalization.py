from __future__ import annotations

import pytest

from flagquantum.compiler.schedule_legalization import (
    ScheduleLegalizationError,
    schedule_circuit_dependencies,
)
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.integration


def _schedule(*instructions: Instruction, max_depth: int | None = None):
    return schedule_circuit_dependencies(
        CircuitIR(3, instructions),
        target_snapshot_id="phase28-target-snapshot",
        max_depth=max_depth,
    )


def test_independent_operations_share_an_asap_layer() -> None:
    schedule = _schedule(Instruction("h", (0,)), Instruction("x", (2,)))

    assert schedule.layers == ((0, 1),)
    assert schedule.depth == 1
    assert schedule.maximum_parallel_width == 2
    assert schedule.dynamic_dependency_count == 0


def test_wire_dependencies_preserve_source_order() -> None:
    schedule = _schedule(
        Instruction("h", (0,)),
        Instruction("x", (0,)),
        Instruction("z", (1,)),
    )

    assert schedule.layers == ((0, 2), (1,))
    assert schedule.instructions[1].predecessors == (0,)
    assert schedule.instructions[1].dependency_kinds == ("wire",)


def test_measurement_feedback_is_a_cross_wire_causal_dependency() -> None:
    schedule = _schedule(
        Instruction(
            "measure",
            (0,),
            metadata={"is_dynamic": True, "classical_bit": 4},
        ),
        Instruction("x", (2,), metadata={"conditions": ((4, 1),)}),
    )

    assert schedule.layers == ((0,), (1,))
    assert schedule.instructions[1].predecessors == (0,)
    assert schedule.instructions[1].dependency_kinds == (
        "barrier",
        "classical",
        "wire",
    )
    assert schedule.dynamic_dependency_count == 1


def test_condition_clauses_collect_all_measurement_producers() -> None:
    schedule = _schedule(
        Instruction(
            "measure",
            (0,),
            metadata={"is_dynamic": True, "classical_bit": 0},
        ),
        Instruction(
            "measure",
            (1,),
            metadata={"is_dynamic": True, "classical_bit": 1},
        ),
        Instruction(
            "x",
            (2,),
            metadata={"condition_clauses": (((0, 1),), ((1, 0),))},
        ),
    )

    assert schedule.layers == ((0,), (1,), (2,))
    assert schedule.instructions[2].predecessors == (0, 1)


def test_classical_read_before_measurement_fails_closed() -> None:
    with pytest.raises(ScheduleLegalizationError, match="before measurement"):
        _schedule(Instruction("x", (1,), metadata={"conditions": ((0, 1),)}))


def test_malformed_conditions_fail_closed() -> None:
    with pytest.raises(ScheduleLegalizationError, match="malformed"):
        _schedule(Instruction("x", (1,), metadata={"conditions": ((0, 2),)}))


def test_schedule_depth_budget_fails_closed() -> None:
    with pytest.raises(ScheduleLegalizationError, match="depth 2 exceeds maximum 1"):
        _schedule(Instruction("h", (0,)), Instruction("x", (0,)), max_depth=1)


def test_schedule_identity_binds_snapshot_and_is_deterministic() -> None:
    circuit = CircuitIR(1, (Instruction("h", (0,)),))
    first = schedule_circuit_dependencies(circuit, target_snapshot_id="snapshot-a")
    second = schedule_circuit_dependencies(circuit, target_snapshot_id="snapshot-a")
    other = schedule_circuit_dependencies(circuit, target_snapshot_id="snapshot-b")

    assert first.program is circuit
    assert first.schedule_identity == second.schedule_identity
    assert first.schedule_identity != other.schedule_identity
    assert len(first.schedule_identity) == 64
