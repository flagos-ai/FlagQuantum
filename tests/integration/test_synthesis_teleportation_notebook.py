"""Verify the native private-feedback teleportation Notebook workflow."""

import cmath
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.quantum_teleportation.comparisons.circuit import (
    measured_branches,
    reference_circuit,
)
from examples.synthesis.algorithms.foundational.quantum_teleportation.overview.workflow import (
    Workflow,
    run_workflow,
)
from flagquantum.compiler.resource_estimation import ResourceEstimate


@pytest.mark.parametrize("theta,phi", ((math.pi / 2, 0.0), (0.73, -0.41), (1.12, 0.37)))
def test_complete_private_feedback_channel(theta: float, phi: float) -> None:
    workflow = run_workflow(theta=theta, phi=phi)
    program = workflow.program
    expected = fq.run(reference_circuit(theta, phi)).statevector().flatten()
    raw = fq.run(program).statevector().reshape(8, -1)
    torch.testing.assert_close(raw[:, 0], expected, atol=1e-12, rtol=0)
    assert float(raw[:, 1:].abs().square().sum()) < 1e-22
    psi = torch.tensor(
        [[math.cos(theta / 2), cmath.exp(1j * phi) * math.sin(theta / 2)]],
        dtype=torch.complex128,
    )
    branches = measured_branches(psi)
    assert branches.abs().square().sum(dim=1).tolist() == pytest.approx([0.25] * 4)
    torch.testing.assert_close(
        workflow.density, branches.T @ branches.conj(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        workflow.density,
        torch.outer(psi.flatten(), psi.flatten().conj()),
        atol=1e-12,
        rtol=0,
    )
    assert sum(count for _, count in workflow.observations) == 1000
    probability_zero = dict(workflow.observations).get(0, 0) / 1000
    assert abs(probability_zero - math.cos(theta / 2) ** 2) < 0.1
    # Retained outcome qubits are not silently claimed to be reset workspace.
    environment = raw[:, 0].reshape(2, 4).abs().square().sum(dim=0)
    torch.testing.assert_close(
        environment, torch.full((4,), 0.25, dtype=torch.float64), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("theta,bit", ((0.0, 0), (math.pi, 1)))
def test_actual_basis_output_needs_no_user_bitstring_decode(
    theta: float, bit: int
) -> None:
    workflow = run_workflow(theta=theta)
    assert workflow.observations == ((bit, 1000),)
    expected = torch.zeros((2, 2), dtype=torch.complex128)
    expected[bit, bit] = 1
    torch.testing.assert_close(workflow.density, expected, atol=1e-12, rtol=0)


def test_sampling_seed_is_explicit_and_repeatable() -> None:
    assert run_workflow().observations == run_workflow().observations
    assert run_workflow(seed=4).observations != run_workflow().observations


@pytest.mark.parametrize(
    "theta,phi",
    (
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (True, 0.0),
        (0.7, float("nan")),
        (0.7, float("inf")),
        (0.7, True),
    ),
)
def test_invalid_angles_refused(theta: float, phi: float) -> None:
    with pytest.raises(ValueError, match="finite real"):
        run_workflow(theta=theta, phi=phi)


@pytest.mark.parametrize(
    "shots,seed",
    ((0, 3), (100001, 3), (True, 3), (1000, -1), (1000, True), (1000, 2**63)),
)
def test_invalid_sampling_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        run_workflow(shots=shots, seed=seed)


pytestmark = pytest.mark.integration


def test_clean_notebook_executes_complete_workflow() -> None:
    from tests.integration.synthesis_notebook_support import execute_code_cells

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/foundational/quantum_teleportation/overview/walkthrough.ipynb"
    )
    notebook = json.loads(path.read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 4
    workflow = namespace["workflow"]
    assert isinstance(workflow, Workflow)
    assert sum(value for _, value in workflow.observations) == 1000
    torch.testing.assert_close(
        workflow.density,
        torch.full((2, 2), 0.5, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "budget", ("width", "workspace", "operations", "depth", "memory")
)
def test_nonzero_parameter_resource_refusals(budget: str) -> None:
    from examples.synthesis.algorithms.foundational.quantum_teleportation.main import (
        main,
    )
    from flagquantum import synth
    from flagquantum.errors import PlanningError

    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(
            main,
            classical_inputs={"theta": 0.73, "phi": -0.41},
            max_qubits=3 if budget == "width" else None,
            max_workspace=0 if budget == "workspace" else None,
            max_operations=11 if budget == "operations" else None,
            max_depth=8 if budget == "depth" else None,
            memory_limit_bytes=1 if budget == "memory" else None,
        )


@pytest.mark.parametrize(
    "theta,phi,width,workspace,operations,depth",
    (
        (math.pi / 2, 0.0, 3, 0, 9, 6),
        (0.73, -0.41, 4, 1, 12, 9),
        (1.12, 0.37, 4, 1, 12, 9),
    ),
)
def test_bound_parameter_resources(
    theta: float, phi: float, width: int, workspace: int, operations: int, depth: int
) -> None:
    workflow = run_workflow(theta=theta, phi=phi)
    resources = workflow.program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == width
    assert resources.n_operations == operations and resources.depth == depth
    assert workflow.program.report["workspace_qubits"] == workspace
