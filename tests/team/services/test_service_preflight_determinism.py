from __future__ import annotations

from copy import deepcopy

import pytest

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.services import preflight_execution


def _program() -> CircuitIR:
    return CircuitIR(n_wires=1, instructions=(Instruction("h", (0,)),))


def test_execution_preflight_is_repeatable_and_does_not_mutate_input() -> None:
    program = _program()
    original = deepcopy(program)
    options = {"require_gradients": False, "target": "full_state"}

    first = preflight_execution(program, **options)
    second = preflight_execution(program, **options)

    assert first.to_dict() == second.to_dict()
    assert first.executable
    assert first.plan is not None
    assert first.plan["n_wires"] == 1
    assert program == original


def test_invalid_program_fails_closed_repeatably() -> None:
    first = preflight_execution(object())
    second = preflight_execution(object())

    assert first.to_dict() == second.to_dict()
    assert not first.executable
    assert first.blockers[0].code == "INVALID_QUANTUM_PROGRAM"


def test_planning_failure_has_a_repeatable_structured_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.runtime import planner

    def fail_planning(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deterministic planner fixture failure")

    monkeypatch.setattr(planner, "plan_runtime_selection", fail_planning)

    first = preflight_execution(_program())
    second = preflight_execution(_program())

    assert first.to_dict() == second.to_dict()
    assert not first.executable
    assert first.validation.valid
    assert first.blockers[0].code == "EXECUTION_PLANNING_FAILED"
    assert first.blockers[0].retryable
    assert first.blockers[0].message == "deterministic planner fixture failure"
