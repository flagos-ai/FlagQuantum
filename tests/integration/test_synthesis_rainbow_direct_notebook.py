"""Check the original fixed-point grid, coherent loading and sampled interval."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method import (
    physics,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.overview import (
    model,
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_model_regenerated_with_quarter_precision() -> None:
    fresh = model.fresh_model()
    assert fresh.offset == Fraction(-5, 4) and fresh.threshold == Fraction(3, 2)
    assert fresh.selectors == (0, 4, 8, 12, 0, 4, 8, 12, 1, 4, 8, 12, 4, 7, 10, 13)
    assert fresh.rate == pytest.approx(physics.EXPONENT_RATE, abs=1e-14)
    assert fresh.angles == pytest.approx(physics.loading_angles(), abs=1e-14)
    assert fresh.lower_angle == pytest.approx(physics.low_angle(), abs=1e-14)
    assert fresh.exact_price == pytest.approx(24.932955958535754, abs=1e-12)


@pytest.mark.parametrize("power", (0, 1))
def test_complete_environment_and_clean_work(power: int) -> None:
    fresh = model.fresh_model()
    program = workflow.build_program(fresh, power)
    result = fq.run(program)
    actual = result.statevector().reshape(1024, -1)
    expected = fq.run(workflow.reference_circuit(fresh, power)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20
    named = program.probabilities(result, registers=("indicator",))
    assert named[(1,)] == pytest.approx(
        float(expected[512:].abs().square().sum()), abs=1e-10
    )


def test_sampling_retains_negative_estimate_and_interval() -> None:
    result = workflow.run_pricing()
    draws = tuple(
        (row.k, row.shots, row.successes) for row in result.estimate.iterations
    )
    assert draws == ((0, 256, 9), (1, 256, 130))
    assert result.price < 0
    assert result.interval[0] <= result.model.exact_price <= result.interval[1]
    assert abs(result.price - model.SOURCE_PRICE) <= result.source_tolerance


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
