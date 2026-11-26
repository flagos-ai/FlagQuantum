"""Verify the original LiH tapering and complete native VQE objective."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.chemistry.lih_chemistry.overview import (
    model,
    symmetry,
    variational,
)
from flagquantum.algorithms.chemistry.molecular import (
    create_molecular_hamiltonian,
    second_quantized_hamiltonian,
)

pytestmark = pytest.mark.integration


def test_tapered_full_lih_and_native_optimizer() -> None:
    native = create_molecular_hamiltonian(
        model.GEOMETRY, basis=model.BASIS, n_active_electrons=2, n_active_orbitals=5
    )
    operator = second_quantized_hamiltonian(
        native.one_body, native.two_body, native.nuclear_repulsion + native.core_energy
    )
    sectors = symmetry.analyze(operator)
    assert len(sectors.hamiltonian.terms) == 276
    assert len(sectors.generators) == 4 and len(sectors.minima) == 16
    assert sectors.spectral_error < 2e-11
    assert sectors.hf_word == 0 and sectors.reduction.n_qubits == 6
    assert native.fci_energy is not None
    sector_minimum = dict(sectors.minima)[sectors.hf_sector]
    assert abs(sector_minimum - native.fci_energy) < 2e-10
    result = variational.fit(sectors)
    assert len(result.excitation_terms) == 10
    assert result.terms == 231 and len(result.probes) <= 200
    assert result.state_error < 3e-12
    assert min(trial.energy for trial in result.probes) < result.probes[0].energy
    assert all(torch.isfinite(torch.tensor(trial.energy)) for trial in result.probes)


def test_source_stages_are_complete_and_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/chemistry/lih_chemistry/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 23
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
