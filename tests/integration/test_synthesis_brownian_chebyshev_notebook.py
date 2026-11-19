"""Verify both arithmetic demonstrations and all retained price branches."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_complete_brownian_state_and_named_marginal() -> None:
    result = workflow.run()
    assert result.nonzero_words == 32
    assert len(result.distribution) == 24
    assert abs(sum(result.distribution.values()) - 1) < 1e-12
    assert result.state_error < 1e-12
    assert result.workspace < 1e-24
    assert result.price_resources.n_qubits <= 139
    assert result.price_resources.n_qubits == 23
    assert result.sine[0] == Fraction(-1, 4)
    assert result.recurrence[-1] == (
        Fraction(3, 4),
        Fraction(1),
        Fraction(3, 2),
        Fraction(1, 8),
        Fraction(-45, 32),
    )


def test_fresh_gaussians_and_quantized_input_compatibility() -> None:
    fine = model.gaussian(4)
    assert len(fine.points) == len(fine.probabilities) == 16
    assert fine.points[0] == -3
    assert all(
        abs(a - b) < 1e-12
        for a, b in zip(fine.probabilities, reversed(fine.probabilities), strict=True)
    )
    assert abs(sum(fine.probabilities) - 1) < 1e-12
    assert model.gaussian(1).probabilities == (0.5, 0.5)
    with pytest.raises(ValueError, match="symmetric"):
        quantum.build_price((0.51, 0.49))


def test_original_fourteen_stages_are_unexecuted() -> None:
    overview = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/finance/brownian_chebyshev_polynomials/overview"
    )
    cells = json.loads((overview / "walkthrough.ipynb").read_text())["cells"]
    proof = json.loads((overview / "provenance.json").read_text())
    stages = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 14
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
