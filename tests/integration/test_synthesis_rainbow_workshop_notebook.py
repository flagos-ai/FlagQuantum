"""Check the workshop model reuse and the original finite-shot experiment."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview.model import (
    ALPHA,
    EPSILON,
    SOURCE_PRICE,
    STRIKE,
    fresh_model,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview.workflow import (
    reference_circuit,
)
from examples.synthesis.tutorials.workshops.finance_workshops.rainbow_options_workshop_bruteforce.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("budget", (14, 25))
def test_source_preparation_full_state_and_original_bound(budget: int) -> None:
    model = fresh_model()
    program = workflow.build_program(model, max_width=budget)
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference_circuit(model, 0)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert set(program.registers) == {"indicator"}
    assert workflow.maximum_asset_price(model) >= STRIKE


def test_real_20000_shots_meet_source_precision() -> None:
    result = workflow.run_pricing()
    assert 0 < ALPHA < 1
    assert workflow.SHOTS == 20_000
    assert result.probability == result.successes / workflow.SHOTS
    assert result.amplitude_interval[1] - result.amplitude_interval[0] <= 2 * EPSILON
    assert (
        result.price_interval[0] <= result.model.exact_price <= result.price_interval[1]
    )
    assert abs(result.price - SOURCE_PRICE) <= result.source_tolerance


def test_source_stages_are_unexecuted() -> None:
    assert workflow.__file__ is not None
    path = Path(workflow.__file__).parent
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 21
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
