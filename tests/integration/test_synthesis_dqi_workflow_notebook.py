"""Full-state, finite sampling and fresh source model checks for DQI."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.data import (
    PARITY_COLUMNS,
    SYNDROME_TABLE,
    WEIGHTS,
)
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.overview.model import (
    dicke_amplitudes,
    majority_errors,
    make_model,
)
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.overview.quantum import (
    build_dicke,
)
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.overview.workflow import (
    run_workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("weight, unary", ((4, False), (2, True)))
def test_absolute_dicke_state(weight: int, unary: bool) -> None:
    actual = fq.run(build_dicke(weight, use_unary=unary)).statevector().flatten()
    expected = torch.tensor(dicke_amplitudes(weight), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_fresh_tables_full_state_and_observed_shots() -> None:
    result = run_workflow()
    model = make_model()
    assert model.columns == PARITY_COLUMNS
    assert model.syndrome_table == SYNDROME_TABLE
    assert model.weights == pytest.approx(WEIGHTS, abs=1e-14, rel=0)
    assert majority_errors(model) == 12 and sum(model.objective) == 0
    assert result.unary == ((255, 2048),)
    assert len(result.dicke_four) == len(result.dicke_two) == 15
    assert (
        sum(count for _, count in result.dicke_four)
        == sum(count for _, count in result.dicke_two)
        == 2048
    )
    assert sum(count for _, count in result.dqi) == 10000
    assert all(y == 0 for (y, _), _ in result.dqi)
    assert result.mean_score == pytest.approx(4, abs=0.08)
    actual = fq.run(result.program).statevector().reshape(4096, -1)
    expected = fq.run(reference_circuit(model.weights)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


def test_complete_unexecuted_source_stages() -> None:
    from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.overview import (
        __file__ as path,
    )

    assert path is not None
    root = Path(path).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 15
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
