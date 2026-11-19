"""Validate all thermal links, both mock bindings and the full-model subroutine."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.automotive.cooling_systems_optimization.overview import (
    full_model,
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_fresh_thermal_and_mock_workflows() -> None:
    result = workflow.run()
    assert result.configurations.shape == (64, 4)
    assert result.relative_error < 0.04
    assert result.matrix_error < 1e-10 and result.conditional_error < 1e-10
    assert result.reciprocal.fit.phases.numel() == 12
    assert result.inverse_error < 1e-10
    assert result.reference[-1] < 0 and result.computed[-1] < 0
    assert result.mock_resources.n_qubits == 8
    assert result.mock_probabilities.shape == (2, 64)
    assert max(result.mock_errors) < 1e-10
    torch.testing.assert_close(
        result.mock_probabilities.sum(dim=1), torch.ones(2, dtype=model.REAL)
    )
    assert result.conditional_resources.n_qubits == 13


def test_full_design_subroutine_preserves_binding_and_thermal_projection() -> None:
    import flagquantum as fq

    reciprocal = model.inverse_target(1 / 3, 11, target_scale=0.3)
    solver = full_model.conditional_solver(reciprocal)
    assert solver.n_qubits == 15
    heat = torch.tensor(model.HEAT, dtype=model.REAL)
    for word in (11, 63):
        initial = torch.zeros(1 << solver.n_qubits, dtype=model.COMPLEX)
        initial[word * (1 << full_model.SOLVER_BITS)] = 1
        bound = fq.Circuit.from_ir(solver.to_ir(), dtype=model.COMPLEX, inputs=initial)
        state = bound.state().reshape(64, -1)
        projected = state[word, :4]
        expected = model.spectral_solution(
            model.build_matrix(word), reciprocal, heat / heat.norm()
        )
        torch.testing.assert_close(
            -projected, expected.to(model.COMPLEX), atol=1e-10, rtol=0
        )
    assert full_model.WIDTH == 20


def test_original_cells_and_original_mock_table_mapping() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/automotive/cooling_systems_optimization/overview"
    )
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    stages = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(provenance["reviewed_code_cells"]) == 23
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
    assert provenance["original_assets"][0]["native_path"] == "data/costs.json"
    assert abs(quantum.QAE_PHASE_WEIGHT - 0.0980171403295606) < 1e-12
