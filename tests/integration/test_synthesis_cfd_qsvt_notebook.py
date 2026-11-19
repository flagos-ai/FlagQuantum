"""Check both original QSVT degrees and the symmetric fit derivative."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview import (
    phases,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("large,degree,data", [(False, 101, 3), (True, 255, 4)])
def test_source_inverse(large: bool, degree: int, data: int) -> None:
    result = workflow.solve(large)
    assert result.request_degree == degree and result.data_width == data
    assert result.phases.phases.numel() == degree + 1
    assert torch.equal(result.phases.phases, result.phases.phases.flip((0,)))
    assert result.phases.iterations <= 12
    assert result.phases.residual < 1e-10
    assert result.spectral_error < 1e-10
    assert result.full_state_error < 1e-10
    assert result.solution_l2 < 0.1
    assert result.resources.n_qubits == 9
    assert 0 < result.projection_probability < 1


def test_phase_jacobian_matches_finite_difference() -> None:
    x = torch.tensor([0.16, 0.47, 0.92], dtype=torch.float64)
    half = torch.tensor([0.13, -0.17, 0.24], dtype=torch.float64)
    values, derivative = phases.values_and_derivative(phases.signal(x), half)
    torch.testing.assert_close(
        values,
        phases.response(x, torch.cat((half, half.flip((0,))))),
        atol=1e-13,
        rtol=0,
    )
    for i in range(half.numel()):
        delta = torch.zeros_like(half)
        delta[i] = 1e-6
        a, b = half + delta, half - delta
        plus = phases.response(x, torch.cat((a, a.flip((0,)))))
        minus = phases.response(x, torch.cat((b, b.flip((0,)))))
        numerical = (plus - minus) / 2e-6
        torch.testing.assert_close(derivative[:, i], numerical, atol=1e-9, rtol=0)


def test_all_source_stages() -> None:
    folder = Path(workflow.__file__).parent
    notebook = json.loads((folder / "walkthrough.ipynb").read_text())
    proof = json.loads((folder / "provenance.json").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 14
    assert all(c["execution_count"] is None for c in code)
