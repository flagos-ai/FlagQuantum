"""Selected candidates meet actual Runtime costs, phases, cleanup and gradients."""

import cmath
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis_selection import polynomial, pressure
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import select_arithmetic
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.runtime.execution_plan_contract import plan_execution_program
from flagquantum.services._synthesis_execution import inspect_execution

pytestmark = pytest.mark.integration
BASIS = ("x", "h", "s", "sdg", "t", "tdg", "ry", "rz", "cx", "u3", "swap")


def _options():
    return fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )


def _target(routed):
    return (
        Target(
            basis=BASIS,
            couplings=tuple(
                CouplingMap(n, tuple((i, i + 1) for i in range(n - 1))) for n in (9, 10)
            ),
            optimize=False,
            restore_layout=False,
        )
        if routed
        else Target()
    )


def _assert_prediction(plan, product):
    # Runtime owns the canonical sealed program and removes runtime_config from
    # program metadata; Compiler retains its explicit compilation configuration.
    # Do not add a duplicate Runtime canonicalizer to production Compiler.
    expected = replace(
        product.executable,
        metadata={
            k: v
            for k, v in product.executable.metadata.items()
            if k != "runtime_config"
        },
    )
    assert plan_execution_program(plan).content_hash == expected.content_hash


def test_selection_example_executes_both_constraint_driven_choices() -> None:
    run = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_selection"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "selected: recompute" in run.stdout and "selected: retain" in run.stdout
    assert "exact phase/clean workspace: passed" in run.stdout


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_selected_nonzero_xor_complex_phase_and_layout_on_real_runtime(
    routed, objective
) -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    prep = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).s(0).t(1).x(2).to_ir()
    target = _target(routed)
    report = select_arithmetic(
        (graph,), entry=graph.name, preparation=prep, target=target, objective=objective
    )
    row = report.selected
    assert row is not None and row.target is not None and row.arithmetic is not None
    assert row.strategy == ("recompute" if objective == "width" else "retain")
    plan = fq.plan(row.target.submitted, options=_options())
    _assert_prediction(plan, row.target)
    checked = inspect_execution(
        plan,
        limits=CircuitLimits(
            row.resources.n_qubits, row.resources.n_operations, row.resources.depth
        ),
        basis=target.basis,
        coupling=row.target.coupling,
        registers=report.registers,
    )
    result = fq.run(checked.plan)
    assert checked.verify_result(result) == row.resources
    state = result.statevector().reshape(-1)
    reference = torch.zeros_like(state)
    n = row.resources.n_qubits
    layout = row.target.layout
    if routed:
        assert layout.logical_to_physical != tuple(range(n))
        assert any(g.name == "swap" for g in row.target.executable.instructions)
    for x in range(2):
        for y in range(2):
            logical = (
                (((x << 1) | y) << 2) | (2 ^ pressure(x, y))
            ) << row.arithmetic.peak_workspace
            physical = sum(
                ((logical >> (n - 1 - q)) & 1) << (n - 1 - layout.physical_of(q))
                for q in range(n)
            )
            reference[physical] = (1j**x) * cmath.exp(1j * math.pi * y / 4) / 2
    torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)
    dirty = torch.zeros_like(state.real, dtype=torch.bool)
    indices = torch.arange(state.numel())
    for q in range(4, n):
        dirty |= ((indices >> (n - 1 - layout.physical_of(q))) & 1).bool()
    assert float(state[dirty].abs().square().sum()) <= 1e-12
    with pytest.raises(ValueError, match="max_operations"):
        inspect_execution(
            plan,
            limits=CircuitLimits(max_operations=row.resources.n_operations - 1),
            basis=target.basis,
            coupling=row.target.coupling,
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_all_small_inputs_and_all_xor_outputs_selected_without_manual_gates(
    objective,
) -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    for x in range(2):
        for y in range(2):
            for output in range(4):
                value = ((x << 1) | y) << 2 | output
                # Expert seam preparation; CS06-C owns the named normal facade.
                prep = fq.Circuit(4, dtype=torch.complex128)
                for q in range(4):
                    if value & (1 << (3 - q)):
                        prep.x(q)
                report = select_arithmetic(
                    (graph,),
                    entry=graph.name,
                    preparation=prep.to_ir(),
                    objective=objective,
                )
                row = report.selected
                plan = fq.plan(row.target.submitted, options=_options())
                checked = inspect_execution(plan, registers=report.registers)
                result = fq.run(checked.plan)
                checked.verify_result(result)
                state = result.statevector().reshape(-1)
                reference = torch.zeros_like(state)
                expected = (
                    (((x << 1) | y) << 2) | (output ^ pressure(x, y))
                ) << row.arithmetic.peak_workspace
                reference[expected] = 1
                torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize("named_basis", [False, True])
@pytest.mark.parametrize("theta", [0.37, 1.1, 2.0])
def test_polynomial_differentiable_preparation_has_exact_nonzero_gradient(
    named_basis, theta
) -> None:
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    graph = capture(polynomial, {"x": 1, "y": 1})
    prep = fq.Circuit(2, dtype=torch.complex128).ry(0, parameter).to_ir()
    report = select_arithmetic(
        (graph,),
        entry=graph.name,
        preparation=prep,
        target=Target(basis=BASIS) if named_basis else Target(),
    )
    row = report.selected
    assert row is not None
    result = fq.run(row.target.submitted, options=_options())
    _assert_prediction(result.plan, row.target)
    checked = inspect_execution(
        result.plan, basis=report.target.basis, registers=report.registers
    )
    checked.verify_result(result)
    state = result.statevector().reshape(-1)
    indices = torch.arange(state.numel())
    q = checked.registers["result"][-1]
    signs = 1 - 2 * ((indices >> (row.resources.n_qubits - 1 - q)) & 1)
    expectation = (state.abs().square() * signs).sum()
    gradient = torch.autograd.grad(expectation, parameter)[0]
    torch.testing.assert_close(expectation, torch.cos(parameter), atol=1e-10, rtol=0)
    torch.testing.assert_close(gradient, -torch.sin(parameter), atol=1e-8, rtol=0)
    assert torch.isfinite(gradient) and abs(float(gradient)) > 0.1


def test_cost_example_reproduces_full_matrix_and_current_manual_comparison() -> None:
    import json

    run = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_selection_costs"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    rows = json.loads(run.stdout)
    assert len(rows) == 34
    assert len([r for r in rows if r["strategy"] == "manual"]) == 2
    assert all(
        r["construction_seconds"] >= 0 and r["target_seconds"] >= 0 for r in rows
    )
    for name in ("logical", "named_basis"):
        group = [
            r
            for r in rows
            if r["function"] == "polynomial"
            and r["inputs"] == {"x": 2, "y": 2}
            and r["target"] == name
        ]
        assert {r["strategy"] for r in group} == {"retain", "recompute", "manual"}
        assert all(r["qubits"] == 18 and r["workspace"] == 10 for r in group)
        assert (
            next(r for r in group if r["strategy"] == "manual")["operations"]
            > next(r for r in group if r["selected"])["operations"]
        )


def test_manual_baseline_and_selected_polynomial_share_real_coherent_outputs() -> None:
    from examples.synthesis_selection_costs import manual_polynomial

    manual = manual_polynomial()
    prep = fq.Circuit(4, dtype=torch.complex128)
    for q in range(4):
        prep.h(q)
    graph = capture(polynomial, {"x": 2, "y": 2})
    selection = select_arithmetic((graph,), entry=graph.name, preparation=prep.to_ir())
    row = selection.selected
    manual_circuit = fq.Circuit(18, dtype=torch.complex128)
    for q in range(4):
        manual_circuit.h(q)
    manual_circuit.compose(manual)
    for program in (manual_circuit.to_ir(), row.target.submitted):
        state = fq.run(program, options=_options()).statevector().reshape(-1)
        reference = torch.zeros_like(state)
        for x in range(4):
            for y in range(4):
                reference[((((x << 2) | y) << 4) | (x * x + y)) << 10] = 0.25
        torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)
