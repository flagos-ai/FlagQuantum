"""Check the fresh Gaussian QSVT phases and complete amplified state."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_size_fresh_phases_and_exact_amplification() -> None:
    result = workflow.run()
    assert result.even.phases.numel() == 29
    assert result.odd.phases.numel() == 28
    assert result.continuum_overlap == pytest.approx(0.39579583583, abs=1e-10, rel=0)
    assert result.measured_overlap == pytest.approx(0.39191102410, abs=1e-9, rel=0)
    assert 0.003 < result.continuum_overlap - result.measured_overlap < 0.005
    assert max(result.fit_error) < 2e-6
    assert max(result.even.residual, result.odd.residual) < 1e-10
    assert max(result.prep_error, result.amplified_error) < 1e-10
    assert result.work_probability < 1e-22
    assert result.accepted_mass == pytest.approx(1.0, abs=1e-10, rel=0)
    assert result.distribution.shape == (256,)
    assert result.mse < 3e-11
    assert bool(torch.isfinite(result.distribution).all())
    assert result.resources.n_qubits == result.target_resources.n_qubits == 18
    assert set(result.target_resources.operation_counts) == {"cx", "u3"}


def test_source_stages_are_unexecuted() -> None:
    directory = Path(workflow.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    proof = json.loads((directory / "provenance.json").read_text())
    code = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 12
    assert all(c["execution_count"] is None for c in code)
