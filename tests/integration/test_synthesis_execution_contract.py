"""Real execution, phase/layout checks and source-versus-executable refusals."""

import cmath
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis_workspace import candidate, pressure
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.compiler.layout import remove_layout_restore
from flagquantum.core.ir import CircuitIR, MeasurementNode
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.execution_plan_contract import plan_execution_program
from flagquantum.runtime.measurements import execute_measurements
from flagquantum.services._synthesis_execution import (
    ExecutionContractError,
    inspect_execution,
)

pytestmark = pytest.mark.integration

TOPOLOGY = CouplingMap(3, ((0, 1), (1, 2)))


def test_execution_contract_example_runs_as_documented() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_execution_contract"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "runtime basis drift rejected: execution.basis" in result.stdout
    assert "'input': (1,), 'result': (2,)" in result.stdout
    assert "qubits=3 operations=3 depth=3" in result.stdout
    assert "nonidentity routing and full complex phase: passed" in result.stdout


def _options(precision: str = "complex128") -> fq.ExecutionOptions:
    return fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision=precision,
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )


def _source() -> CircuitIR:
    return remove_layout_restore(
        compile_program(
            fq.Circuit(3, dtype=torch.complex128).h(0).s(0).cx(0, 2),
            coupling_map=TOPOLOGY,
            routing_strategy="persistent_layout",
            optimize=False,
        )
    )


@pytest.mark.parametrize(
    "route", ["checked_plan", "normal_source", "serialized_plan", "serialized_source"]
)
def test_real_nonidentity_layout_and_full_complex_phase(route: str) -> None:
    source = _source()
    plan = fq.plan(source, options=_options())
    checked = inspect_execution(
        plan,
        basis=("u3", "swap", "cx"),
        coupling=TOPOLOGY,
        limits=CircuitLimits(max_qubits=3, max_operations=3, max_depth=3),
        registers={"input": (0,), "result": (2,)},
    )
    assert any(g.name == "swap" for g in plan_execution_program(plan).instructions)
    if route == "checked_plan":
        result = fq.run(checked.plan)
    elif route == "normal_source":
        result = fq.run(source, options=_options())
    elif route == "serialized_plan":
        result = fq.run(ExecutionPlan.from_json(plan.to_json()))
    else:
        result = fq.run(CircuitIR.from_json(source.to_json()), options=_options())
    assert checked.verify_result(result) == checked.resources
    state = result.statevector().reshape(-1)
    expected = torch.zeros_like(state)
    # Nonidentity output placement (1,0,2), with a relative phase of i.
    expected[0] = 1 / math.sqrt(2)
    expected[3] = 1j / math.sqrt(2)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert checked.registers == {"input": (1,), "result": (2,)}
    ports = (*checked.registers["input"], *checked.registers["result"])
    measured = execute_measurements(
        state.reshape(1, -1), (MeasurementNode("probabilities", ports),), n_qubits=3
    )[0].value
    torch.testing.assert_close(
        measured.reshape(-1),
        torch.tensor([0.5, 0, 0, 0.5], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_planning_basis_drift_is_refused_before_execution() -> None:
    source = fq.Circuit(1, dtype=torch.complex128).h(0).s(0).to_ir()
    assert {g.name for g in source.instructions} == {"h", "s"}
    plan = fq.plan(source, options=_options())
    assert {g.name for g in plan_execution_program(plan).instructions} == {"u3"}
    with pytest.raises(ExecutionContractError, match="actual executable.*u3"):
        inspect_execution(plan, basis=("h", "s"))
    # A different explicitly requested target is a new check, not silent widening.
    inspected = inspect_execution(plan, basis=("h", "s", "u3"))
    assert inspected.verify_result(fq.run(inspected.plan)).n_operations == 1


def test_invalid_basis_names_and_missing_conversion_rules_are_distinct() -> None:
    source = fq.Circuit(1).h(0).to_ir()
    with pytest.raises(BasisConversionError, match="unknown|undeclared"):
        inspect_execution(
            fq.plan(source, options=_options()), basis=("imaginary_gate",)
        )
    with pytest.raises(BasisConversionError, match="h.*no verified decomposition"):
        convert_basis(source, gates=("u3", "cx"))


def test_source_only_resource_check_would_wrongly_accept_routing_overhead() -> None:
    source = fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 2).to_ir()
    assert estimate_resources(source).n_operations == 2
    routed = remove_layout_restore(
        compile_program(
            source,
            coupling_map=TOPOLOGY,
            routing_strategy="persistent_layout",
            optimize=False,
        )
    )
    plan = fq.plan(routed, options=_options())
    with pytest.raises(ExecutionContractError, match="max_operations=2, actual=3"):
        inspect_execution(
            plan, coupling=TOPOLOGY, limits=CircuitLimits(max_operations=2)
        )
    repaired = inspect_execution(
        plan, coupling=TOPOLOGY, limits=CircuitLimits(max_operations=3)
    )
    assert repaired.verify_result(fq.run(repaired.plan)).n_operations == 3


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_preparation_arithmetic_and_cleanup_are_one_executable_scope(
    strategy: str,
) -> None:
    block = candidate(1, strategy)
    n = block.circuit.n_wires
    source = (
        fq.Circuit(n, dtype=torch.complex128).h(0).h(1).s(0).t(1).compose(block.circuit)
    )
    plan = fq.plan(source, options=_options())
    checked = inspect_execution(
        plan,
        limits=CircuitLimits(max_qubits=n),
        basis=("u3", "cx", "ccx", "h", "s", "t", "x"),
        registers={"x": (0,), "y": (1,), "result": (2, 3)},
    )
    assert (
        checked.resources.n_operations > estimate_resources(block.circuit).n_operations
    )
    with pytest.raises(ExecutionContractError, match="max_operations"):
        inspect_execution(
            plan,
            limits=CircuitLimits(
                max_operations=estimate_resources(block.circuit).n_operations
            ),
        )
    state = fq.run(checked.plan).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    for x in range(2):
        for y in range(2):
            index = (((x << 1) | y) << 2 | pressure(x, y)) << block.peak_workspace
            expected[index] = cmath.exp(1j * (x * math.pi / 2 + y * math.pi / 4)) / 2
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert float(state.reshape(16, -1)[:, 1:].abs().square().sum()) <= 1e-12


def test_wrong_program_options_and_missing_plans_do_not_reuse_old_checks() -> None:
    source = fq.Circuit(1, dtype=torch.complex128).h(0)
    checked = inspect_execution(fq.plan(source, options=_options()))
    good = fq.run(checked.plan)
    for result in (
        fq.run(fq.Circuit(1).x(0), options=_options()),
        fq.run(source, options=replace(_options(), seed=17)),
        replace(good, plan=None),
    ):
        with pytest.raises(ExecutionContractError, match="execution.identity"):
            checked.verify_result(result)
    assert checked.verify_result(good).n_qubits == 1


@pytest.mark.parametrize("damage", ["missing", "dtype", "batch", "width"])
def test_materialized_state_shape_and_precision_must_match_the_plan(
    damage: str,
) -> None:
    checked = inspect_execution(fq.plan(fq.Circuit(1).h(0), options=_options()))
    result = fq.run(checked.plan)
    state = result.statevector()
    values = {
        "missing": None,
        "dtype": state.to(torch.complex64),
        "batch": state.repeat(2, 1),
        "width": state.reshape(2, 1),
    }
    with pytest.raises(ExecutionContractError, match="execution.result"):
        checked.verify_result(replace(result, state=values[damage]))


def test_post_route_conversion_without_intact_swap_evidence_fails_closed() -> None:
    routed = remove_layout_restore(
        compile_program(
            fq.Circuit(3).h(0).cx(0, 2),
            coupling_map=TOPOLOGY,
            routing_strategy="persistent_layout",
            optimize=False,
        )
    )
    converted = convert_basis(routed, gates=("h", "cx"), max_added_operations=100)
    plan = fq.plan(converted.program, options=_options())
    with pytest.raises(ExecutionContractError, match="SWAP evidence"):
        inspect_execution(plan, coupling=TOPOLOGY)


def test_checked_plan_execution_and_verification_do_not_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = inspect_execution(
        fq.plan(_source(), options=_options()), coupling=TOPOLOGY
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("checked-plan path must not replan or recompile")

    import flagquantum.compiler as compiler
    import flagquantum.runtime.planner as planner

    monkeypatch.setattr(fq, "plan", forbidden)
    monkeypatch.setattr(compiler, "compile", forbidden)
    monkeypatch.setattr(planner, "compile_program", forbidden)
    result = fq.run(checked.plan)
    monkeypatch.setattr(fq, "run", forbidden)
    assert checked.verify_result(result).n_operations == 3
