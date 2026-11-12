"""Fresh-register sources share the existing checked Program and named reader."""

import math
from dataclasses import replace
from typing import Any, Literal, NoReturn
from unittest.mock import Mock

import pytest

import flagquantum as fq
from flagquantum import _api
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.selection import Constraints, ConstructionBudget
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.services._synthesis_registers import synthesize_registers
from flagquantum.services.synthesis import Program, SynthesisError
from tests.integration.test_construction_register_execution import BASIS
from tests.register_program_cases import tutorial_program
from tests.synthesis_workflow_helpers import MEMORY, check_state, native

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("producer", ["native", "recorded"])
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_both_producers_share_program_execution_and_named_reading(
    producer: Literal["native", "recorded"],
    routed: bool,
    objective: Literal["width", "depth"],
) -> None:
    if producer == "native":
        imported = parse_qmod(
            "qfunc main(output x:qnum<2>, output y:qnum) { allocate(x); hadamard_transform(x); y=x**2+1; }",
            source_version=SUPPORTED_SOURCE_VERSION,
        )
        model, functions = imported.program, imported.functions
    else:
        model, functions = tutorial_program(2)
    target = Target(
        basis=BASIS if routed else None,
        couplings=tuple(CouplingMap.line(n) for n in range(1, 21)) if routed else (),
        optimize=not routed,
        restore_layout=not routed,
    )
    program = synthesize_registers(
        model, functions, target=target, objective=objective, memory_limit_bytes=MEMORY
    )
    assert type(program) is Program
    selected = program.selection.selected
    assert selected is not None and selected.target is not None
    assert program.report["source_identities"] == selected.source_identities
    result = fq.run(program)
    check_state(program, result, {(x, x * x + 1): 0.5 for x in range(4)})
    observed = program.probabilities(result, registers=("y", "x"))
    assert dict(observed) == pytest.approx(
        {(x * x + 1, x): 0.25 for x in range(4)}, abs=1e-12
    )
    assert program.report["resources"] == program._checked.verify_result(result)
    if routed:
        assert any(op.name == "swap" for op in selected.target.executable.instructions)


def test_native_phase_and_later_mutation_survive_services_admission() -> None:
    program = native(
        "qfunc main(output flag:qbit, output n:qnum, output res:qnum) { allocate(flag); H(flag); S(flag); allocate(2,n); hadamard_transform(n); res=flag*n; X(flag); }"
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {
            (1 - q, n, q * n): (1j if q else 1) / math.sqrt(8)
            for q in range(2)
            for n in range(4)
        },
    )
    assert dict(program.probabilities(result, registers=("res",))) == pytest.approx(
        {(0,): 5 / 8, (1,): 1 / 8, (2,): 1 / 8, (3,): 1 / 8}
    )


def test_synthesis_plans_once_and_never_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = Mock(wraps=_api.plan)
    monkeypatch.setattr(_api, "plan", plan)
    monkeypatch.setattr(
        _api, "run", Mock(side_effect=AssertionError("synthesis must not execute"))
    )
    program = native("qfunc main(output q:qbit) { allocate(q); X(q); }")
    assert program.execution is program.execution
    assert plan.call_count == 1


def test_constraints_and_incomplete_search_refuse_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_plan(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("infeasible or incomplete synthesis must not plan")

    monkeypatch.setattr(_api, "plan", no_plan)
    source = "qfunc main(output x:qnum<2>, output y:qnum) { allocate(x); y=x*x+1; }"
    with pytest.raises(SynthesisError) as error:
        native(source, constraints=Constraints(CircuitLimits(max_qubits=2)))
    assert error.value.repairs
    with pytest.raises(SynthesisError) as error:
        native(source, budget=ConstructionBudget(candidate_checks=1))
    assert not error.value.selection.complete


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.5])
def test_native_admission_requires_valid_explicit_memory_ceiling(limit: Any) -> None:
    imported = parse_qmod(
        "qfunc main(output q:qbit) { allocate(q); }",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    with pytest.raises((TypeError, ValueError)):
        synthesize_registers(imported.program, memory_limit_bytes=limit)


def test_runtime_capacity_refuses_before_allocation() -> None:
    imported = parse_qmod(
        "qfunc main(output x:qnum<3>, output y:qnum) { allocate(x); y=x*x+1; }",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    with pytest.raises(ValueError, match="memory|Memory|budget"):
        synthesize_registers(imported.program, imported.functions, memory_limit_bytes=1)


def test_named_budget_reaches_existing_marginal_reader_above_eight_bits() -> None:
    program = native("qfunc main(output number:qnum<9>) { number=7; }")
    result = fq.run(program)
    with pytest.raises(ValueError, match="budget"):
        program.probabilities(result, registers=("number",), max_bits=8)
    observed = program.probabilities(result, registers=("number",), max_bits=9)
    assert dict(observed) == {(7,): 1.0}
    assert observed.diagnostics["possible_keys"] == 512


def test_named_reader_keeps_identity_budgets_normalization_and_layout_checks() -> None:
    program = native(
        "qfunc main(output a:qnum, output b:qnum, output res:qnum) { a=2; b=3; res=a*b; }"
    )
    result = fq.run(program)
    with pytest.raises(ValueError, match="register"):
        program.probabilities(result)
    for limits in ({"max_bits": 1}, {"max_keys": 1}, {"max_amplitudes": 1}):
        with pytest.raises(ValueError, match="budget"):
            program.probabilities(result, registers=("res",), **limits)
    with pytest.raises(ValueError, match="identity"):
        program.probabilities(fq.run(fq.Circuit(1)), registers=("res",))
    with pytest.raises(ValueError, match="registers changed"):
        replace(program, registers={"res": (0,)}).execution
    assert result.state is not None
    result.state.mul_(2)
    with pytest.raises(ValueError, match="normalized"):
        program.probabilities(result, registers=("res",))
