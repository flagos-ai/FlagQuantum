"""Check all signed grid branches and the original 12-qubit LCHS calculation."""

import json
import math
from pathlib import Path

import pytest
import torch

np = pytest.importorskip("numpy")

from examples.synthesis.algorithms.quantum_differential_equations_solvers.lchs.overview import (  # noqa: E402
    model,
    phasing,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_reconstruct_original_grid_and_degree() -> None:
    grid = model.source_grid()
    assert len(set(grid.signed)) == 64
    assert grid.signed == tuple(range(32)) + tuple(range(-32, 0))
    assert grid.points[32] == pytest.approx(-grid.radius, abs=1e-12)
    assert grid.points[1] == pytest.approx(grid.step, abs=1e-12)
    assert torch.equal(
        torch.from_numpy(grid.points),
        grid.step * torch.tensor(grid.signed, dtype=torch.float64),
    )
    assert float(grid.kernel_amplitudes.square().sum()) == pytest.approx(1, abs=1e-12)
    assert phasing.degree_for(1 + grid.radius) == 18
    assert phasing.degree_for(16.5, 1e-7) == 33


@pytest.mark.parametrize("alternate", [False, True])
def test_complete_complex_native_state_and_unadjusted_projection(
    alternate: bool,
) -> None:
    grid = model.source_grid()
    initial = model.source_input(4)
    if alternate:
        initial = initial.flip(0)
    run = workflow.compare_quantum(grid, initial)
    expected = reference.dense_state(grid, initial, run.factorization)
    torch.testing.assert_close(run.logical, expected, rtol=0, atol=1e-12)
    assert run.logical.numel() == 4096
    assert run.circuit.n_qubits == 12
    assert len(run.factorization.rows) == 37
    assert len(run.factorization.coefficients) == 37
    assert run.absolute_error < 1e-5
    assert run.quadrature_fidelity > 1 - 1e-10
    assert run.exact_fidelity > 1 - 0.02001
    assert len(run.amplitudes) == 4


@pytest.mark.parametrize("time", [0.0, math.inf, math.nan, 19.0])
def test_phase_degree_profile_refuses_outside_bounds(time: float) -> None:
    with pytest.raises(ValueError, match="finite duration"):
        phasing.degree_for(time)


def test_notebook_has_all_clean_source_stages() -> None:
    root = Path(workflow.__file__).parent
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 16
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
