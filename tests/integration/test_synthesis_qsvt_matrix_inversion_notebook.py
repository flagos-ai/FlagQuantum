"""Check the original dense model and both fresh QSVT inverse routes."""

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("numpy")


def test_both_original_inverse_paths() -> None:
    from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.overview import (
        workflow,
    )

    fresh, manual, automatic = workflow.run_workflow()
    assert fresh.matrix.shape == (4, 4)
    assert fresh.encoding.pauli_counts == (16, 16)
    assert (fresh.encoding.data_width, fresh.encoding.block_width) == (2, 4)
    assert fresh.encoding.alpha == pytest.approx(1.5893797441, abs=1e-9, rel=0)
    assert fresh.effective_condition == pytest.approx(12.8823354019, abs=1e-8, rel=0)
    assert manual.degree == 151
    assert automatic.degree == 95
    assert manual.response_error < workflow.MANUAL_ERROR
    assert (
        automatic.response_error * fresh.condition / workflow.SCALE
        < workflow.AUTOMATIC_ERROR
    )
    for result in (manual, automatic):
        assert result.phase_fit.phases.numel() == result.degree + 1
        assert result.phase_fit.residual < 1e-10
        assert result.full_state_error < 1e-10
        assert result.spectral_error < 1e-10
        assert result.solution_error < 0.05
        assert 0 < result.projection_probability < 1


def test_changed_real_input_and_failures() -> None:
    from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.overview import (
        model,
        workflow,
    )

    fresh = model.build_model()
    rhs = torch.tensor([0.7, -0.11, 0.3, 0.9], dtype=torch.float64)
    alternate = workflow.solve(fresh, rhs, workflow.manual_degree(fresh))
    assert alternate.full_state_error < 1e-10
    assert alternate.spectral_error < 1e-10
    assert alternate.solution_error < 0.05
    with pytest.raises(ValueError, match="finite amplitudes"):
        workflow.solve(fresh, torch.zeros(4), 151)
    with pytest.raises(ValueError, match="odd degree"):
        workflow.solve(fresh, rhs, 150)


def test_source_stages_are_complete_and_unexecuted() -> None:
    from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.overview import (
        workflow,
    )

    folder = Path(workflow.__file__).parent
    notebook = json.loads((folder / "walkthrough.ipynb").read_text())
    proof = json.loads((folder / "provenance.json").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 15
    assert all(c["execution_count"] is None for c in code)
