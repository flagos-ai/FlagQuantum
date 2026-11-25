"""Check all eight original Slater modes and the complete paired Fock state."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_state_preparation.fermionic_gaussian.overview import (
    model,
    quantum,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_source_ring_and_four_mode_elimination() -> None:
    _, occupied, residual = workflow.demo()
    assert occupied.shape == (4, 2) and residual < 1e-13
    ring = model.periodic_hamiltonian()
    assert ring.shape == (8, 8)
    assert ring[0, 6] == ring[1, 7] == -1 and ring[4, 4] == -10
    prepared = workflow.run_slater()
    state = fq.run(prepared.program).statevector().flatten()
    expected = torch.zeros(256, dtype=torch.complex128)
    expected[1 << torch.arange(8)] = prepared.orbital
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert prepared.program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("even", [False, True])
def test_pairing_relative_phase_and_oracle(even: bool) -> None:
    h, delta = model.pairing_inputs()
    if even:
        h = 0.1 * torch.eye(2, dtype=torch.complex128)
        delta = (1 + 0.4j) * delta
    decomposition = model.canonicalize(h, delta)
    vacuum, residual = model.vacuum(decomposition)
    assert residual < 1e-12
    program = quantum.pairing_program(vacuum)
    observed = fq.run(program).statevector().flatten()
    spectral = reference.fock_hamiltonian(h, delta, reference.annihilation(2))
    energies, states = torch.linalg.eigh(spectral)
    ground = states[:, 0]
    overlap = torch.vdot(ground, observed)
    torch.testing.assert_close(observed, ground * overlap, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        spectral @ observed, energies[0] * observed, atol=1e-12, rtol=0
    )
    indices = torch.tensor([1, 2] if even else [0, 3])
    assert float(observed[indices].abs().square().sum()) < 1e-24
    assert program.report["workspace_qubits"] == 0


def test_reject_invalid_symmetries_and_gap() -> None:
    h, delta = model.pairing_inputs()
    with pytest.raises(ValueError, match="antisymmetric"):
        model.canonicalize(h, delta + torch.eye(2))
    with pytest.raises(ValueError, match="gap"):
        model.canonicalize(torch.zeros_like(h), torch.zeros_like(delta))


def test_source_stage_mapping() -> None:
    folder = Path(model.__file__).parent
    book = json.loads((folder / "walkthrough.ipynb").read_text())
    proof = json.loads((folder / "provenance.json").read_text())
    stages = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 25
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
