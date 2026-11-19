"""Run source electron counts, sampled-time grids, and analytic comparisons."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth

from .model import (
    example_matrix,
    expected_state,
    factor,
    hamiltonian,
    reconstruct_orbitals,
)
from .quantum import build_program

SITES = 4
MODES = 8


@dataclass(frozen=True)
class Orbitals:
    energies: torch.Tensor
    vectors: torch.Tensor
    reconstruction_error: float


@dataclass(frozen=True)
class Trajectory:
    program: synth.Program
    states: torch.Tensor
    charge: torch.Tensor
    spin: torch.Tensor
    modes: torch.Tensor
    times: torch.Tensor


def diagonalize(matrix: torch.Tensor, occupied: int) -> Orbitals:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    vectors = eigenvectors[:, :occupied]
    plan = factor(vectors)
    reconstructed = reconstruct_orbitals(plan, matrix.shape[0])
    return Orbitals(eigenvalues, vectors, float((vectors - reconstructed).abs().max()))


def demo_orbitals() -> Orbitals:
    matrix = example_matrix()
    count = int((torch.linalg.eigvalsh(matrix) < 0).sum())
    return diagonalize(matrix, count)


def mode_densities(states: torch.Tensor) -> torch.Tensor:
    """Compute exact expectation of each occupation from all basis weights."""
    words = torch.arange(1 << MODES)
    masks = ((words[:, None] >> torch.arange(MODES)) & 1).to(torch.float64)
    return states.abs().square() @ masks


def evolve(
    vectors: torch.Tensor, times: int, tau: float, hopping: float, contact: float
) -> Trajectory:
    if type(times) is not int or not 1 <= times <= 11:
        raise ValueError("source time grid must contain between one and eleven samples")
    plan = factor(vectors)
    states = []
    for step in range(times):
        program = build_program(plan, step, tau * hopping, tau * contact)
        states.append(fq.run(program).statevector().flatten())
    state_table = torch.stack(states)
    modes = mode_densities(state_table)
    up, down = modes[:, :SITES], modes[:, SITES:]
    return Trajectory(
        program,
        state_table,
        up + down,
        up - down,
        modes,
        torch.arange(times, dtype=torch.float64) * tau,
    )


def initial_one_electron() -> tuple[Trajectory, torch.Tensor, float]:
    orbitals = diagonalize(hamiltonian(3.0, 0.5), 1)
    trajectory = evolve(orbitals.vectors, 1, 0.1, 1.0, 0.0)
    expected = expected_state(orbitals.vectors)
    overlap = torch.vdot(expected, trajectory.states[0])
    return trajectory, expected, float(overlap.abs().square())


def interacting() -> Trajectory:
    orbitals = diagonalize(hamiltonian(3.0, 0.5), 2)
    return evolve(orbitals.vectors, 10, 0.1, 1.0, 3.0)


def spread(trajectory: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    # The final source plot uses (N + 1)/2 with zero-based site labels.
    distances = (torch.arange(SITES, dtype=torch.float64) - (SITES + 1) / 2).abs()
    return trajectory.charge @ distances, trajectory.spin @ distances


def localized() -> torch.Tensor:
    matrix = torch.zeros((MODES, MODES), dtype=torch.float64)
    matrix[0, 0] = -1.0
    return diagonalize(matrix, 1).vectors


def noninteracting() -> Trajectory:
    return evolve(localized(), 11, 0.1, 1.0, 0.0)


def vanishing_hopping() -> Trajectory:
    orbitals = diagonalize(hamiltonian(10.0, 4 / 6), 4)
    return evolve(orbitals.vectors, 11, 0.3, 0.0, 2.0)


def one_particle_curves(
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare numerical diagonalization with exact open-chain standing waves."""
    energies, vectors = torch.linalg.eigh(hamiltonian(0.0, 1.0))
    modes = torch.arange(1, SITES + 1, dtype=torch.float64)
    dispersion = -2 * torch.cos(math.pi * modes / (SITES + 1))
    basis = math.sqrt(2 / (SITES + 1)) * torch.sin(
        math.pi * torch.outer(modes, modes) / (SITES + 1)
    )
    analytic = (
        torch.exp(-1j * torch.outer(times, dispersion)) * basis[:, 0]
    ) @ basis.to(torch.complex128)
    numerical = (
        torch.exp(-1j * torch.outer(times, energies)) * vectors[0]
    ) @ vectors.T.to(torch.complex128)
    torch.testing.assert_close(
        energies, dispersion.repeat(2).sort().values, atol=1e-14, rtol=0
    )
    numerical_occupations = numerical[:, :SITES].abs().square()
    analytic_occupations = analytic.abs().square()
    torch.testing.assert_close(
        numerical_occupations, analytic_occupations, atol=1e-14, rtol=0
    )
    return dispersion, numerical_occupations, analytic_occupations
