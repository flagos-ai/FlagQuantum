"""Validate every native binding-curve point and independent variational states."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.molecular_energy_curve.main import (
    build_program,
)
from examples.synthesis.applications.chemistry.molecular_energy_curve.overview import (
    model,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def points() -> tuple[workflow.Point, ...]:
    return workflow.run_workflow()


def test_complete_source_grid_and_energy_result(
    points: tuple[workflow.Point, ...],
) -> None:
    expected = (0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.75, 2.1, 2.45, 2.8, 3.15, 3.5)
    assert tuple(row.distance for row in points) == pytest.approx(expected)
    for row in points:
        assert row.converged
        assert 0 < row.iterations <= workflow.ITERATION_LIMIT
        assert row.evaluations > 1
        assert row.resources.n_qubits == 1
        assert row.model.coupling_to_complement < 1e-10
        assert row.energy == pytest.approx(row.model.fci, abs=1e-9, rel=0)
        assert row.model.rhf >= row.energy - 1e-12
        assert float(row.model.hamiltonian[1, 1].real) == pytest.approx(
            row.model.rhf, abs=1e-10, rel=0
        )


@pytest.mark.parametrize(("distance", "angle"), [(0.2, 0.43), (3.5, -0.61)])
def test_independent_full_state_and_expectation(distance: float, angle: float) -> None:
    candidate = model.build_model(distance)
    reference = fq.Circuit(1, dtype=torch.complex128).x(0).ry(0, theta=-angle)
    expected = fq.run(reference).statevector().flatten()
    actual = fq.run(build_program((angle,))).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    a, b = expected[0], expected[1]
    h = candidate.hamiltonian
    energy = (
        a.abs().square() * h[0, 0]
        + b.abs().square() * h[1, 1]
        + 2 * (a.conj() * h[0, 1] * b).real
    ).real
    assert workflow.evaluate(candidate, angle) == pytest.approx(
        float(energy), abs=1e-12, rel=0
    )


def test_source_book_keeps_all_stages() -> None:
    directory = Path(model.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    stages = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(provenance["reviewed_code_cells"]) == 5
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in stages
    )
