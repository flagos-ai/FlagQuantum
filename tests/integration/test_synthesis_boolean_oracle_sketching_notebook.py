"""Check every stage of the Boolean oracle sketch and its correction."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("numpy")


def test_original_scale_and_independent_quantum_results() -> None:
    from examples.synthesis.algorithms.quantum_primitives.quantum_oracle_sketching_boolean.overview import (
        model,
        quantum,
        workflow,
    )

    result = workflow.run()
    benchmark = result.benchmark
    assert list(benchmark.table) == [1, 0, 1, 0, 1, 1, 1, 1]
    assert benchmark.demo_samples.size == 200 and benchmark.bell_samples.size == 500
    assert benchmark.sizes.size == 12 and benchmark.sizes[-1] == 31622
    assert model.TRIALS == 1500
    assert benchmark.single_error == pytest.approx(0.16944264428, abs=1e-10, rel=0)
    assert benchmark.mean_error[-1] < benchmark.mean_error[0]
    assert benchmark.mean_operator_error[-1] < benchmark.mean_operator_error[0]
    assert max(result.demo_error, result.known_error, result.bell_error) < 1e-12
    assert (
        result.demo.report["workspace_qubits"]
        == result.bell.report["workspace_qubits"]
        == 0
    )
    assert quantum.SHOTS == 10_000
    assert 0.94 < result.bell_sampled < 0.99
    assert result.threshold.phases.numel() == 19
    assert result.threshold_error < 0.162
    assert result.threshold.residual < 1e-10
    assert result.corrected.resources.n_qubits == 4
    assert (
        max(
            result.corrected.oracle_error,
            result.corrected.spectral_error,
            result.corrected.full_state_error,
        )
        < 1e-12
    )
    assert 0.77 < result.corrected.sampled_success < 0.83
    assert result.corrected.sampled_fidelity > result.bell_sampled


def test_original_stages_are_unexecuted() -> None:
    from examples.synthesis.algorithms.quantum_primitives.quantum_oracle_sketching_boolean.overview import (
        workflow,
    )

    directory = Path(workflow.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    proof = json.loads((directory / "provenance.json").read_text())
    stages = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 9
    assert all(cell["execution_count"] is None for cell in stages)
