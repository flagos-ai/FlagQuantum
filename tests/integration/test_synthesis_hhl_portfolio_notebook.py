"""Verify source data, matrix reconstruction and all native HHL amplitudes."""

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.finance.portfolio_optimization_hhl.hhl_portfolio import (
    physics,
)
from examples.synthesis.applications.finance.portfolio_optimization_hhl.hhl_portfolio.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_source_data_layout_and_fresh_hhl_bindings() -> None:
    fresh = model.build_model()
    assert fresh.prices.shape == (251, 4) and fresh.returns.shape == (250, 4)
    assert (fresh.dates[0], fresh.dates[-1]) == (date(2024, 1, 2), date(2024, 12, 30))
    assert model.ASSETS == ("AAPL", "AMZN", "NVDA", "TSLA")
    torch.testing.assert_close(fresh.covariance, torch.cov(fresh.returns.T) * 251)
    assert fresh.encoded_rhs.nonzero().flatten().tolist() == [10, 11]
    assert len(fresh.angles) == 64 and 62 < fresh.condition < 64
    torch.testing.assert_close(
        fresh.evolution,
        torch.tensor(physics.UNITARY, dtype=model.COMPLEX),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        fresh.angles,
        torch.tensor(physics.ANGLES, dtype=model.DTYPE),
        atol=1e-12,
        rtol=0,
    )


def test_native_original_full_state_and_source_fit() -> None:
    result = workflow.run_workflow()
    assert result.absolute_error < 1e-11
    assert 0 < result.postselection_probability < 1
    assert result.relative_error == pytest.approx(0.080218471184, abs=1e-10)
    assert result.weights.shape == (4, 2) and result.values.shape == (251, 2)
    torch.testing.assert_close(result.least_squares, result.direct, atol=1e-12, rtol=0)
    torch.testing.assert_close(result.values[0], torch.ones(2, dtype=model.DTYPE))
    rebuilt = result.model.prices @ result.weights
    torch.testing.assert_close(result.values, rebuilt / rebuilt[0], atol=1e-12, rtol=0)


def test_nonzero_alternate_rhs_preserves_absolute_state_and_work_cleanup() -> None:
    fresh = model.build_model()
    rhs = torch.tensor(
        [(-1) ** word * (word + 1) for word in range(16)], dtype=model.DTYPE
    )
    alternate = replace(fresh, encoded_rhs=rhs / rhs.norm())
    actual = quantum.logical_state(quantum.build_program(alternate))
    assert actual.shape == (2, 16, 64)
    torch.testing.assert_close(
        actual, workflow.reference_state(alternate), atol=1e-11, rtol=0
    )


def test_original_cells_and_price_data_are_pinned() -> None:
    assert model.__file__ is not None
    path = Path(model.__file__).parent
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 28
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    assert (
        hashlib.sha256((path / "prices.csv").read_bytes()).hexdigest()
        == proof["input_data"]["sha256"]
    )
