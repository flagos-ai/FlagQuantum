"""Verify both original QML sections through native quantum execution."""

import cmath
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytest.importorskip("matplotlib")

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.qml.overview.workflow import (
    Workflow,
    ansatz,
    build_classifier,
    build_variational,
    energy,
    plot_workflow,
    probability_gradient,
    probability_zero,
    run_workflow,
    solve_energy,
    train_not,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import (
    execute_code_cells,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


@pytest.mark.parametrize(
    "angles", ((0.0, 0.0, 0.0), (2.03, 0.31, -0.41), (0.73, -1.1, 0.37))
)
def test_absolute_variational_state_and_clean_workspace(
    angles: tuple[float, ...],
) -> None:
    theta, phi, lam = angles
    cosine, sine = math.cos(theta / 2), math.sin(theta / 2)
    matrix = torch.tensor(
        [
            [cosine, -cmath.exp(1j * lam) * sine],
            [cmath.exp(1j * phi) * sine, cmath.exp(1j * (phi + lam)) * cosine],
        ],
        dtype=torch.complex128,
    )
    reference = fq.Circuit(1, dtype=torch.complex128).any(0, matrix=matrix)
    compare_states(build_variational(angles), reference, ("q",))
    state = fq.run(reference).statevector().flatten()
    hamiltonian = torch.tensor([[1.0, -1.0], [-1.0, 0.0]], dtype=torch.complex128)
    expected = float(torch.vdot(state, hamiltonian @ state).real)
    assert energy(angles) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x,weight", ((0.0, 3.0), (math.pi, 3.0), (0.73, -0.41), (1.2, 6.1), (0.0, 0.0))
)
def test_classical_bindings_full_state_readout_and_quantum_gradient(
    x: float, weight: float
) -> None:
    parameter = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    reference = (
        fq.Circuit(1, dtype=torch.complex128).rx(0, theta=x).rx(0, theta=parameter)
    )
    compare_states(build_classifier(x, weight), reference, ("res",))
    expected = fq.run(reference).statevector().flatten()[0].abs().square()
    gradient = torch.autograd.grad(expected, parameter)[0]
    assert probability_zero(x, weight) == pytest.approx(
        float(expected.detach()), abs=1e-12
    )
    assert probability_gradient(x, weight) == pytest.approx(float(gradient), abs=1e-12)
    assert build_classifier(x, weight).report["workspace_qubits"] == 0


def test_every_original_five_epoch_update_matches_independent_autograd(
    workflow: Workflow,
) -> None:
    reference = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([reference], lr=1.0)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert len(workflow.training.steps) == 5
    for row in workflow.training.steps:
        optimizer.zero_grad()
        assert row.weight_before == pytest.approx(float(reference.detach()), abs=1e-12)
        predictions = torch.stack(
            [
                fq.run(
                    fq.Circuit(1, dtype=torch.complex128)
                    .rx(0, theta=x)
                    .rx(0, theta=reference)
                )
                .statevector()
                .flatten()[0]
                .abs()
                .square()
                for x in (math.pi, 0.0)
            ]
        )
        loss = torch.nn.functional.l1_loss(predictions, labels)
        reference.grad = torch.autograd.grad(loss, reference)[0]
        assert reference.grad is not None
        assert row.loss == pytest.approx(float(loss.detach()), abs=1e-12)
        assert row.gradient == pytest.approx(float(reference.grad), abs=1e-12)
        optimizer.step()
        assert row.weight_after == pytest.approx(float(reference.detach()), abs=1e-12)
    assert workflow.training.weight == pytest.approx(
        float(reference.detach()), abs=1e-12
    )
    assert workflow.training.accuracy == 1.0
    torch.testing.assert_close(
        torch.tensor(workflow.training.predictions),
        labels.to(torch.float32),
        atol=1e-2,
        rtol=0,
    )


def test_vqe_matches_independent_hamiltonian_oracle_and_budget(
    workflow: Workflow,
) -> None:
    oracle = torch.linalg.eigvalsh(
        torch.tensor([[1.0, -1.0], [-1.0, 0.0]], dtype=torch.float64)
    )
    actual = workflow.variational
    assert actual.optimizer.converged and actual.optimizer.iterations <= 200
    assert len(actual.trace) == actual.optimizer.evaluations
    assert actual.trace[0] == pytest.approx(1.0, abs=1e-12)
    assert actual.optimizer.value == pytest.approx(float(oracle[0]), abs=1e-10)
    assert all(
        float(oracle[0]) - 1e-12 <= value <= float(oracle[1]) + 1e-12
        for value in actual.trace
    )
    assert energy(
        tuple(float(v) for v in actual.optimizer.parameters)
    ) == pytest.approx(actual.optimizer.value, abs=1e-12)


def test_bounded_vqe_does_not_claim_convergence() -> None:
    result = solve_energy(maxiter=1)
    assert result.optimizer.iterations == 1 and not result.optimizer.converged
    assert len(result.trace) == result.optimizer.evaluations


def test_figures_contain_actual_quantum_costs_and_update_losses(
    workflow: Workflow,
) -> None:
    pytest.importorskip("numpy")
    import numpy as np

    figure = plot_workflow(workflow)
    assert len(figure.axes) == 2
    energy_data = figure.axes[0].lines[0].get_ydata()
    training_data = figure.axes[1].lines[0].get_ydata()
    assert isinstance(energy_data, np.ndarray) and isinstance(training_data, np.ndarray)
    assert energy_data.tolist() == pytest.approx(
        workflow.variational.trace, abs=0, rel=0
    )
    assert training_data.tolist() == pytest.approx(
        [step.loss for step in workflow.training.steps], abs=0, rel=0
    )


def test_clean_notebook_executes_both_complete_sections() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/basics/qml/overview/walkthrough.ipynb"
    )
    notebook = json.loads(path.read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 4
    actual = namespace["workflow"]
    assert isinstance(actual, Workflow)
    assert actual.variational.optimizer.converged
    assert len(actual.training.steps) == 5 and actual.training.accuracy == 1.0


@pytest.mark.parametrize("value", (0, 201, True))
def test_iteration_limits_refuse(value: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        solve_energy(maxiter=value)
    with pytest.raises(ValueError, match="epochs"):
        train_not(epochs=value)


@pytest.mark.parametrize(
    "angles",
    ((0.0,), (0.0,) * 4, (True, 0.0, 0.0), (math.nan, 0.0, 0.0), (0.0, math.inf, 0.0)),
)
def test_invalid_variational_parameters_refuse(angles: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        build_variational(angles)


@pytest.mark.parametrize("value", (True, math.nan, math.inf))
def test_invalid_classical_parameters_refuse(value: float) -> None:
    with pytest.raises(ValueError):
        probability_zero(value, 0.3)
    with pytest.raises(ValueError):
        probability_gradient(0.7, value)
    with pytest.raises(ValueError):
        train_not(initial_weight=value)


@pytest.mark.parametrize("value", (0.0, -1.0, True, math.nan, math.inf))
def test_invalid_learning_rate_refuses(value: float) -> None:
    with pytest.raises(ValueError):
        train_not(learning_rate=value)


@pytest.mark.parametrize("budget", ("operations", "depth", "memory"))
def test_variational_resource_refusals(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(
            ansatz,
            classical_inputs={"angles": [0.73, -1.1, 0.37]},
            max_operations=0 if budget == "operations" else None,
            max_depth=0 if budget == "depth" else None,
            memory_limit_bytes=1 if budget == "memory" else None,
        )


def test_minimal_logical_width_and_zero_workspace_are_accepted() -> None:
    program = synth.synthesize(
        ansatz,
        classical_inputs={"angles": [0.73, -1.1, 0.37]},
        max_qubits=1,
        max_workspace=0,
        max_operations=1,
        max_depth=1,
        memory_limit_bytes=1024**2,
    )
    assert program.circuit.n_wires == 1 and program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("section", ("variational", "training"))
def test_actual_bound_resources(workflow: Workflow, section: str) -> None:
    program = (
        workflow.variational.program
        if section == "variational"
        else workflow.training.program
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == resources.n_operations == resources.depth == 1
    assert program.report["workspace_qubits"] == 0
    assert dict(resources.operation_counts) == (
        {"u3": 1} if section == "variational" else {"rx": 1}
    )
