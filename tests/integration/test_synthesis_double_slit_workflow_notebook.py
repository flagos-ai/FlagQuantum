"""Verify the complete original-resolution wave circuit and source constraint."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment.overview import (
    model,
    quantum,
)
from examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment.overview.workflow import (
    run,
)

pytestmark = pytest.mark.integration


def test_original_width_and_complete_complex_state() -> None:
    result = run()
    assert result.resources.n_qubits == 18
    assert result.resources.max_operation_width <= 7
    assert result.full_error < 1e-10 and result.spectral_error < 1e-10
    assert result.work_probability == 0
    assert result.inverse.phases.phases.numel() == 192
    assert result.wave.numel() == 256
    assert result.fidelity == pytest.approx(result.inverse.expected_fidelity, abs=1e-10)
    assert result.fidelity >= 0.95
    assert result.selected_probability == pytest.approx(0.0457958213, abs=1e-10)


def test_physical_grid_boundaries_holes_and_refused_nonbijection() -> None:
    reference = model.build_model()
    # Left boundary has only the inward gradient and the diagonal term.
    assert int(torch.count_nonzero(reference.matrix[2])) == 2
    # A solid site in the band is removed while both apertures remain open.
    assert int(torch.count_nonzero(reference.matrix[11 * 16 + 6])) == 4
    assert int(torch.count_nonzero(reference.matrix[7 * 16 + 6])) == 1
    assert int(torch.count_nonzero(reference.source)) == 2
    with pytest.raises(ValueError, match="bijective"):
        quantum.permutation(2, lambda _: 0)


def test_one_original_cell_maps_to_eight_unexecuted_sections() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/cfd/double_slit_experiment/quantum_double_slit_experiment/overview"
    )
    book = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 8
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
    (original,) = proof["reviewed_code_cells"]
    assert original["index"] == 1
    assert original["native_code_cells"] == list(range(1, 9))
