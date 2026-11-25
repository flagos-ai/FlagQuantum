"""Check the regenerated model, complete complex state and actual SWAP samples."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.physical_systems.hhl_lanchester.overview import (
    model,
    quantum,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def prepared() -> model.Model:
    return model.build_model()


@pytest.fixture(scope="module")
def solution(prepared: model.Model) -> workflow.Solution:
    return workflow.solve(prepared)


def test_full_fresh_source_model_and_populations(prepared: model.Model) -> None:
    assert prepared.operator.shape == (16, 16)
    assert prepared.padded.shape == (32,)
    assert prepared.hermitian.shape == prepared.evolution.shape == (32, 32)
    assert prepared.angles.numel() == 64
    assert prepared.padded[:16].count_nonzero() == 0
    assert float(prepared.padded.norm()) == pytest.approx(1.0, abs=1e-14)
    direct = torch.linalg.solve(prepared.operator, prepared.forcing)
    recursive = [(30.0, 1.0)]
    for _ in range(7):
        rabbits, foxes = recursive[-1]
        recursive.append(
            (0.94 * rabbits - 0.6 * foxes + 2.4, 0.88 * foxes + 0.12 * rabbits + 0.06)
        )
    expected = torch.tensor(recursive, dtype=torch.float64).T.flatten()
    torch.testing.assert_close(direct, expected, rtol=0, atol=1e-12)


def test_all_absolute_amplitudes_and_signed_result(
    prepared: model.Model, solution: workflow.Solution
) -> None:
    expected = reference.spectral_state(
        prepared.phase_eigenvalues,
        prepared.eigenvectors,
        prepared.padded,
        prepared.angles,
    )
    torch.testing.assert_close(solution.logical, expected, rtol=0, atol=1e-12)
    assert solution.logical.shape == (2, 32, 64)
    assert solution.resources.n_qubits == 16
    assert 0.999 < solution.fidelity <= 1.0
    assert 0.02 < solution.probability < 0.025
    assert float(solution.statevector.imag.abs().max()) < 1e-11


def test_coherent_swap_keeps_indicator_and_phase_branches(
    solution: workflow.Solution,
) -> None:
    actual = quantum.sampled_swap(solution.logical, solution.reference)
    expected = reference.reference_swap(solution.logical, solution.reference)
    torch.testing.assert_close(actual.probabilities, expected, rtol=0, atol=1e-12)
    assert int(actual.counts.sum()) == 30_000
    selected = int(actual.counts[:, 1].sum())
    assert 0 < selected < 30_000
    assert actual.root_overlap**2 == pytest.approx(
        2 * int(actual.counts[0, 1]) / selected - 1
    )


def test_notebook_is_unexecuted_and_maps_all_source_stages() -> None:
    root = Path(workflow.__file__).parent
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 23
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
