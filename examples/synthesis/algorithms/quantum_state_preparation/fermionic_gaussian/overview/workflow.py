"""Check the original Slater and pairing examples with complete native states."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.overview.model import (
    example_matrix,
    factor,
    reconstruct_orbitals,
)
from flagquantum import synth

from .model import Canonical, canonicalize, pairing_inputs, periodic_hamiltonian, vacuum
from .quantum import pairing_program, slater_program
from .reference import annihilation, fock_hamiltonian


@dataclass(frozen=True)
class Slater:
    program: synth.Program
    values: torch.Tensor
    amplitudes: torch.Tensor
    orbital: torch.Tensor
    fidelity: float


@dataclass(frozen=True)
class Paired:
    program: synth.Program
    decomposition: Canonical
    amplitudes: torch.Tensor
    eigenvalues: torch.Tensor
    fidelity: float
    vacuum_residual: float


def demo() -> tuple[torch.Tensor, torch.Tensor, float]:
    matrix = example_matrix()
    values, modes = torch.linalg.eigh(matrix)
    occupied = modes[:, values < 0]
    reconstruction = reconstruct_orbitals(factor(occupied), 4)
    return values, occupied, float((occupied - reconstruction).abs().max())


def run_slater() -> Slater:
    program, energies = slater_program()
    state = fq.run(program).statevector().flatten()
    _, vectors = torch.linalg.eigh(periodic_hamiltonian())
    orbital = vectors[:, 0].to(torch.complex128)
    amplitudes = state[1 << torch.arange(8)]
    fidelity = float(torch.vdot(orbital, amplitudes).abs().square())
    return Slater(program, energies, amplitudes, orbital, fidelity)


def run_pairing() -> Paired:
    h, delta = pairing_inputs()
    decomposition = canonicalize(h, delta)
    vector, residual = vacuum(decomposition)
    program = pairing_program(vector)
    state = fq.run(program).statevector().flatten()
    eigenvalues, eigenvectors = torch.linalg.eigh(
        fock_hamiltonian(h, delta, annihilation(2))
    )
    fidelity = float(torch.vdot(eigenvectors[:, 0], state).abs().square())
    return Paired(program, decomposition, state, eigenvalues, fidelity, residual)
