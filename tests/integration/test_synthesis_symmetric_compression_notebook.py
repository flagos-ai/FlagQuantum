"""Check all three native compression outputs and finite-shot observations."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.overview.quantum import (
    SOURCE_WEIGHTS,
    Scenario,
    adjacent_rotation,
    endpoint_rotation,
    source_amplitudes,
)
from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.overview.reference import (
    reference,
)
from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.overview.workflow import (
    SCENARIOS,
    SHOTS,
    run_case,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_full_complex_state_and_source_shots(scenario: Scenario) -> None:
    result = run_case(scenario, 67)
    expected = fq.run(reference(scenario)).statevector().flatten()
    torch.testing.assert_close(result.state, expected, atol=2e-12, rtol=0)
    assert sum(result.counts) == SHOTS == 2048
    assert math.fsum(result.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert all(
        count == 0
        for probability, count in zip(result.probabilities, result.counts, strict=True)
        if probability == 0.0
    )
    if scenario == "uniform":
        assert len(result.probabilities) == 32
        assert result.probabilities[:6] == pytest.approx(
            tuple(math.comb(5, weight) / 32 for weight in range(6))
        )
        assert not any(result.counts[6:])
    elif scenario == "one_excitation":
        assert result.counts == (0, 2048, 0, 0, 0, 0, 0, 0)
    else:
        assert result.probabilities == pytest.approx(
            (1 / 8, 1 / 24, 1 / 24, 1 / 8, 1 / 12, 1 / 12, 1 / 4, 1 / 4)
        )


def test_ratios_and_independent_unitarity() -> None:
    assert math.fsum(weight**2 for weight in SOURCE_WEIGHTS) == pytest.approx(1 / 3)
    amplitudes = source_amplitudes()
    assert math.fsum(weight**2 for weight in amplitudes) == pytest.approx(1.0)
    assert tuple(index for index, weight in enumerate(amplitudes) if weight) == (
        1,
        2,
        4,
    )
    for size in (3, 4, 5):
        factors = [adjacent_rotation(size, position) for position in range(1, size - 1)]
        for factor in (*factors, endpoint_rotation(size)):
            matrix = torch.tensor(factor, dtype=torch.float64)
            torch.testing.assert_close(
                matrix.T @ matrix, torch.eye(8, dtype=torch.float64), atol=1e-14, rtol=0
            )


def test_original_stage_provenance() -> None:
    from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    path = Path(overview_path).parent
    proof = json.loads((path / "provenance.json").read_text())
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 12
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
