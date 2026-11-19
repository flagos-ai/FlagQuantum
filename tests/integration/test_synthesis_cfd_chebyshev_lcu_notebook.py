"""Verify all original Chebyshev-LCU solves with independent references."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "large,tolerance,degree,max_error,physical",
    [(False, 0.0, 31, 0.2, 12), (True, 0.0, 255, 0.1, 15), (True, 0.03, 255, 0.1, 15)],
)
def test_original_solve(
    large: bool, tolerance: float, degree: int, max_error: float, physical: int
) -> None:
    result = workflow.solve(large, tolerance)
    assert result.degree == degree
    assert result.encoding.data_width == (4 if large else 3)
    assert result.selector_width == (7 if large else 4)
    assert result.resources.n_qubits == physical
    assert result.encoding.pauli_counts == (None if large else (24, 20))
    assert result.resources.operation_counts["any"] == result.selector_width + 1
    assert result.full_state_error < 1e-10
    assert result.projected_reference_error < 1e-10
    assert result.preparation_error <= tolerance + 1e-12
    assert 0 < result.projection_probability < 1
    assert result.solution_error == pytest.approx(
        float((result.expected - result.solution).norm()), abs=1e-11
    )
    assert result.solution_error < max_error
    assert bool(torch.isfinite(result.solution).all())


def test_refuse_non_source_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        workflow.solve(False, 0.03)
    with pytest.raises(ValueError, match="tolerance"):
        workflow.solve(True, 0.01)


def test_all_source_stages() -> None:
    folder = Path(workflow.__file__).parent
    notebook = json.loads((folder / "walkthrough.ipynb").read_text())
    proof = json.loads((folder / "provenance.json").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 16
    assert all(c["execution_count"] is None for c in code)
