"""Check the source affine conversion and the complete native integration state."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.overview.model import (
    EPSILON,
    FRACTION_BITS,
    SOURCE_PRICE,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method import (
    physics,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


def test_source_affine_conversion_every_joint_selector_and_inclusive_endpoint() -> None:
    model = workflow.build_model()
    asset = model.asset_model
    assert len(model.weights) == 32 and sum(model.weights) == pytest.approx(1)
    angles, low = physics.classical_parameters(asset.rate)
    assert model.angles == pytest.approx(angles, abs=1e-14)
    assert model.lower_angle == pytest.approx(low, abs=1e-14)
    threshold = asset.threshold * (1 << FRACTION_BITS)
    lower_probability = math.sin(model.lower_angle / 2) ** 2
    for score in asset.selectors:
        p = (
            math.fsum(model.weights[: score + 1])
            if score > threshold
            else lower_probability
        )
        assert model.price_offset + model.price_scale * p == pytest.approx(
            asset.payoff[score], abs=1e-10
        )


def test_original_preparation_full_state_and_clean_work() -> None:
    model = workflow.build_model()
    program = workflow.build_program(model, 0)
    result = fq.run(program)
    actual = result.statevector().reshape(1024, -1)
    expected = fq.run(workflow.reference_circuit(model, 0)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20
    named = program.probabilities(
        result, registers=("indicator",), max_amplitudes=1 << workflow.MAX_WIDTH
    )
    assert named[(1,)] == pytest.approx(
        float(expected[512:].abs().square().sum()), abs=1e-10
    )


def test_actual_sampling_and_source_interval() -> None:
    pricing = workflow.run_pricing()
    draws = tuple(
        (row.k, row.shots, row.successes) for row in pricing.estimate.iterations
    )
    assert draws == ((0, 256, 4), (1, 256, 119))
    lo, hi = pricing.estimate.confidence_interval
    assert hi - lo <= 2 * EPSILON
    assert (
        pricing.interval[0]
        <= pricing.model.asset_model.exact_price
        <= pricing.interval[1]
    )
    assert abs(pricing.price - SOURCE_PRICE) <= pricing.source_tolerance


def test_source_stages_are_unexecuted() -> None:
    assert workflow.__file__ is not None
    path = Path(workflow.__file__).parent
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 16
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
