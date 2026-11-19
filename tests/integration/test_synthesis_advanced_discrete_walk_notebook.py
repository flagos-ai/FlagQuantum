"""Original discrete walk distributions and the same-basis width sweep."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.overview import (
    sweep,
    walks,
)

pytestmark = pytest.mark.integration


def test_original_quantum_and_classical_walks() -> None:
    data = walks.run()
    for row, width in ((data.circle, 8), (data.balanced, 8), (data.cube, 7)):
        assert row.resources.n_qubits == width
        assert sum(row.counts) == walks.SHOTS
        assert math.isclose(sum(row.exact), 1, abs_tol=1e-12)
        assert row.maximum_error < 1e-11 and row.work_probability < 1e-24
    for row in (data.circle, data.balanced):
        assert row.positions == tuple(range(-64, 64))
        assert all(
            count == 0
            for position, count in zip(row.positions, row.counts, strict=True)
            if position % 2
        )
    assert data.circle.counts != data.balanced.counts
    assert sum(data.classical_circle) == sum(data.classical_cube) == walks.SHOTS
    assert data.cube.exact[-1] == pytest.approx(9 / 16, abs=1e-12)
    assert abs(data.cube.observed(15) - data.cube.exact[-1]) < 0.025


def test_original_width_limits_and_exact_cx_candidate_selection() -> None:
    rows = sweep.sweep()
    assert tuple(row.width_limit for row in rows) == tuple(range(12, 21))
    for route in (
        tuple(row.arithmetic for row in rows),
        tuple(row.cascade for row in rows),
    ):
        counts = tuple(sweep.cost(candidate)[0] for candidate in route)
        assert counts == tuple(sorted(counts, reverse=True))
        assert counts[0] > counts[-1] > 0
        for row, candidate in zip(rows, route, strict=True):
            assert set(candidate.resources.operation_counts) == {"u3", "cx"}
            assert candidate.resources.n_qubits <= row.width_limit
    # The complete endpoint simulations also run in the Notebook/CLI workflow.
    assert sweep.validate(rows[-1].arithmetic) < 1e-10


def test_source_size_and_width_bound() -> None:
    with pytest.raises(ValueError, match="12 through 20"):
        sweep.select("arithmetic", 11, {})
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/tutorials/advanced_tutorials/discrete_quantum_walk/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 19
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
