"""Check fresh H2 coefficients, coherent molecular walks and finite samples."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.overview import (
    model,
    quantum,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("optimized", [False, True])
def test_complete_walk_and_clean_workspace(optimized: bool) -> None:
    fresh = model.build_model()
    expected = fq.run(reference.circuit(fresh, optimized)).statevector().reshape(-1)
    program = (
        quantum.optimized_program(fresh) if optimized else quantum.naive_circuit(fresh)
    )
    actual = fq.run(program).statevector().reshape(8192, 16)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20


def test_full_sampled_workflow_and_shared_target_basis() -> None:
    result = workflow.run_workflow()
    assert len(result.model.coefficients) == 15
    assert result.model.probabilities[-1] == 0
    assert sum(result.model.probabilities) == pytest.approx(1.0, abs=1e-12, rel=0)
    assert result.model.max_reference_delta < 1e-7
    assert result.model.full_ci == pytest.approx(-1.13730603575, abs=1e-8, rel=0)
    for method in (result.naive, result.optimized):
        assert sum(method.block_zero_counts) == method.selected_count < workflow.SHOTS
        assert method.selected_count > 0
        assert method.signed_phases[15:17] == (15 / 32, -0.5)
        assert (
            method.estimate
            == method.energies[max(range(32), key=method.block_zero_counts.__getitem__)]
        )
        assert abs(method.estimate - result.model.full_ci) < result.error_tolerance
        assert method.resources.n_qubits == 17
        assert set(method.resources.operation_counts) == {"cx", "u3"}
    assert result.optimized.resources.depth < result.naive.resources.depth


def test_all_source_stages() -> None:
    directory = Path(workflow.__file__).parent
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    proof = json.loads((directory / "provenance.json").read_text())
    stages = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 27
    assert all(cell["execution_count"] is None for cell in stages)
