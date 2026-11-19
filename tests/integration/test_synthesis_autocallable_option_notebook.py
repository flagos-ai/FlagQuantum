"""Complete autocallable financial and coherent execution acceptance."""

import json
import math
from collections import Counter
from pathlib import Path

import pytest

import examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.overview.model as model
import examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.overview.workflow as workflow
from flagquantum import synth

pytestmark = pytest.mark.integration


def test_original_grid_paths_and_encoding() -> None:
    grid = model.gaussian_grid()
    transform = model.encoding(grid)
    rows = model.paths(grid)
    assert grid.points == (-3.0, -1.5, 0.0, 1.5)
    assert grid.probabilities == pytest.approx(
        (0.065634503010069, 0.434365496989931, 0.434365496989931, 0.065634503010069),
        abs=1e-15,
    )
    assert (grid.raw_offset, grid.raw_slope, transform.width) == (-2, 1, 4)
    assert len(rows) == len({row.words for row in rows}) == 64
    assert math.fsum(row.probability for row in rows) == pytest.approx(1.0)
    assert Counter(row.payment for row in rows) == {
        model.Payment.FIRST_CALL: 16,
        model.Payment.SECOND_CALL: 4,
        model.Payment.PUT: 33,
        model.Payment.ZERO: 11,
    }
    assert model.comparisons(grid, transform) == (2, 4, 6, 1, 3, 5, 2)
    reference = transform.reference_probabilities()
    for row in rows:
        if row.payment is model.Payment.PUT:
            threshold = row.log_returns[-1] + 8
            probability = math.fsum(reference[: threshold + 1])
        else:
            probability = math.sin(transform.rotation(row.discounted_payoff) / 2) ** 2
        assert transform.decode(probability) == pytest.approx(
            row.discounted_payoff, abs=1e-12
        )
    # The first log return exactly at the barrier does not trigger it.
    strict = next(row for row in rows if row.words == (1, 3, 2))
    assert strict.log_returns == (-1, 0, 0)
    assert strict.payment is model.Payment.ZERO


def test_complete_states_and_decoded_price() -> None:
    result = workflow.run()
    assert result.expectation == pytest.approx(-3.5032073946209374, abs=1e-12)
    assert result.exact_price == pytest.approx(result.expectation, abs=1e-12)
    preparation, grover = result.checks
    assert (preparation.iterations, grover.iterations) == (0, 1)
    theta = math.asin(math.sqrt(preparation.probability))
    assert grover.probability == pytest.approx(math.sin(3 * theta) ** 2, abs=1e-12)
    for check in result.checks:
        assert check.resources.n_qubits == 22
        assert check.resources.n_operations <= 5000
        assert check.state_error < 1e-12
        assert check.workspace_probability < 1e-24


def test_budget_and_disabled_source_stage() -> None:
    grid = model.gaussian_grid()
    transform = model.encoding(grid)
    with pytest.raises(synth.SynthesisError, match="qubits"):
        workflow.build_program(grid, transform, 0, max_qubits=21)
    with pytest.raises(ValueError, match="0 to 4"):
        workflow.build_program(grid, transform, 5)
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/finance/autocallable_options/quantum_autocallable_option_pricing/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 36
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
    assert "".join(code[33]["source"]).startswith("# Original disabled run")
