"""Execute fresh dense and native LCHS comparisons at the source sizes."""

from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.phases import (
    Factorization,
)

from .model import (
    DISCRETIZATION_ERROR,
    DTYPE,
    GQSP_ERROR,
    KERNEL_ERROR,
    SCALE,
    TIME,
    Grid,
    exact_solution,
    quadrature_solution,
    quantum_operators,
    source_input,
)
from .phasing import fit_source
from .quantum import build_program

CLASSICAL_SIZE = 128
CLASSICAL_TIME = 10
CLASSICAL_SEEDS = (1930, 1931)


def fidelity(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Compare the complex directions of two nonzero vectors."""
    return float(
        torch.vdot(actual / actual.norm(), reference / reference.norm()).abs().square()
    )


def _random_complex(size: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    shape = (size, size)
    return torch.from_numpy(rng.normal(size=shape) + 1j * rng.normal(size=shape))


def classical_reference(grid: Grid) -> float:
    """Regenerate the source 128-dimensional PSD and Hermitian comparison."""
    raw_h = _random_complex(CLASSICAL_SIZE, CLASSICAL_SEEDS[0])
    raw_l = _random_complex(CLASSICAL_SIZE, CLASSICAL_SEEDS[1])
    h = (raw_h + raw_h.mH) / 2
    ell = raw_l.mH @ raw_l
    h = h / torch.linalg.svdvals(h).max()
    ell = ell / torch.linalg.svdvals(ell).max()
    initial = source_input(CLASSICAL_SIZE)
    expected = exact_solution(initial, h, ell, CLASSICAL_TIME)
    mixture = quadrature_solution(initial, h, ell, CLASSICAL_TIME, grid)
    return fidelity(mixture, expected)


@dataclass(frozen=True)
class QuantumComparison:
    """Absolute complex projection, full state, and actual synthesis resources."""

    circuit: fq.Circuit
    factorization: Factorization
    logical: torch.Tensor
    amplitudes: torch.Tensor
    normalized_quadrature: torch.Tensor
    exact_fidelity: float
    quadrature_fidelity: float
    absolute_error: float


def compare_quantum(grid: Grid, initial: torch.Tensor) -> QuantumComparison:
    """Execute the coherent six-bit kernel and verify both source references."""
    phases = fit_source(grid)
    circuit = build_program(grid, initial, phases)
    logical: torch.Tensor = fq.run(circuit).statevector().flatten()
    amplitudes = logical.reshape(64, 8, 4, 2)[0, 0, :, 0]
    h, ell = quantum_operators()
    quadrature = quadrature_solution(initial, h, ell, TIME, grid)
    normalized = quadrature * SCALE / grid.kernel_weights.abs().sum()
    exact = exact_solution(initial, h, ell, TIME)
    absolute_error = float((amplitudes - normalized).norm())
    exact_fidelity = fidelity(amplitudes, exact)
    if absolute_error >= GQSP_ERROR:
        raise ValueError(
            "Native complex projection exceeds the GQSP discretized error bound"
        )
    if 1 - exact_fidelity > GQSP_ERROR + KERNEL_ERROR + DISCRETIZATION_ERROR:
        raise ValueError(
            "Native evolution exceeds the combined source fidelity tolerance"
        )
    return QuantumComparison(
        circuit,
        phases,
        logical,
        amplitudes.to(DTYPE),
        normalized,
        exact_fidelity,
        fidelity(amplitudes, normalized),
        absolute_error,
    )
