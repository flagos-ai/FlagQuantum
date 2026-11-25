"""Check source regeneration, exact amplitudes and finite-shot pricing."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method import (
    physics,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview import (
    model,
    workflow,
)

pytestmark = pytest.mark.integration


def test_fresh_source_payoff_and_original_parameters() -> None:
    fresh = model.fresh_model()
    assert fresh.coefficients == (
        (Fraction(1), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2)),
    )
    assert fresh.offset == Fraction(-3, 2)
    assert fresh.selectors == (0, 2, 4, 6) * 4
    assert fresh.probabilities == pytest.approx(physics.PROBABILITIES, abs=1e-14)
    assert fresh.angles == pytest.approx(physics.effective_angles(), abs=1e-14)
    assert fresh.exact_price == pytest.approx(model.SOURCE_PRICE, abs=1e-4)
    assert fresh.payoff_grid[0] == model.STRIKE


@pytest.mark.parametrize("power,width", [(0, 14), (1, 25), (2, 25)])
def test_native_absolute_state_against_independent_circuit(
    power: int, width: int
) -> None:
    fresh = model.fresh_model()
    program = workflow.build_program(fresh, power, width)
    result = fq.run(program)
    state = result.statevector().flatten()
    expected = fq.run(workflow.reference_circuit(fresh, power)).statevector().flatten()
    assert state.numel() == expected.numel() == 32
    torch.testing.assert_close(state, expected, atol=1e-11, rtol=0)
    distribution = program.probabilities(result, registers=("indicator",))
    assert distribution[(1,)] == pytest.approx(
        float(expected[16:].abs().square().sum()), abs=1e-11
    )


def test_actual_sample_counts_and_confidence_rule() -> None:
    result = workflow.run_pricing()
    draws = tuple(
        (row.k, row.shots, row.successes) for row in result.estimate.iterations
    )
    assert draws == ((0, 256, 152), (1, 256, 143))
    assert abs(result.price - model.SOURCE_PRICE) <= result.source_tolerance
    assert result.interval[0] <= result.model.exact_price <= result.interval[1]
    assert (
        result.estimate.confidence_interval[1] - result.estimate.confidence_interval[0]
        <= 2 * model.EPSILON
    )


def test_source_stages_are_unexecuted() -> None:
    assert model.__file__ is not None
    path = Path(model.__file__).parent
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 15
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
