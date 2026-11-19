"""Regenerate source grid, Hamiltonians, and nonunitary references."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

DTYPE = torch.complex128
TIME = 1.0
SHIFT = 2.0
GQSP_ERROR = 1e-5
KERNEL_ERROR = 1e-2
DISCRETIZATION_ERROR = 1e-2
GRID_BITS = 6
STATE_BITS = 2
GRID_WORDS = 1 << GRID_BITS
SCALE = 0.99
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """Exact source radius and grid data indexed as native signed words."""

    gamma: float
    radius: float
    step: float
    signed: tuple[int, ...]
    points: RealArray
    kernel_amplitudes: torch.Tensor
    kernel_weights: torch.Tensor


def source_grid() -> Grid:
    """Derive the original grid dimensions, rather than loading sample values."""
    gamma = math.sqrt(SHIFT + math.log((1 + 1 / (2 * math.pi)) / KERNEL_ERROR)) / SHIFT
    radius = 2 * SHIFT * gamma**2
    estimate = math.pi / (
        TIME / 2 + math.log(64 * math.exp(3 * SHIFT / 2) / (15 * DISCRETIZATION_ERROR))
    )
    bits = math.ceil(math.log2(2 * radius / estimate))
    if bits != GRID_BITS:
        raise ValueError("The reconstructed LCHS grid must use six bits")
    second_half = GRID_WORDS // 2
    signed = tuple(range(second_half)) + tuple(range(-second_half, 0))
    step = radius / second_half
    k = np.asarray(signed, dtype=np.float64) * step
    unnormalized = torch.from_numpy(
        np.exp(-(k**2) / (8 * gamma**2)) / np.sqrt(1 + k**2)
    )
    weight = np.exp(-1j * k * SHIFT) / (1 + k**2) * np.exp(-(k**2 + 1) / (4 * gamma**2))
    return Grid(
        gamma,
        radius,
        step,
        signed,
        k,
        unnormalized / unnormalized.norm(),
        torch.from_numpy(weight),
    )


def quantum_operators() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the original data Hamiltonian H and positive operator L."""
    id2 = torch.eye(2, dtype=DTYPE)
    x = torch.tensor(((0, 1), (1, 0)), dtype=DTYPE)
    z = torch.diag(torch.tensor((1, -1), dtype=DTYPE))
    return (torch.kron(x, x) + torch.kron(z, z)) / 2, (
        torch.eye(4, dtype=DTYPE) + torch.kron(id2, z)
    ) / 2


def source_input(size: int) -> torch.Tensor:
    """Reproduce the source RandomState(1) input without touching its global RNG."""
    vector = torch.tensor(np.random.RandomState(1).rand(size), dtype=torch.float64)
    return torch.div(vector, vector.norm())


def exact_solution(
    state: torch.Tensor, h: torch.Tensor, ell: torch.Tensor, time: float
) -> torch.Tensor:
    """Evaluate the independent dense nonunitary evolution."""
    result: torch.Tensor = torch.linalg.matrix_exp(-time * (ell + 1j * h)) @ state.to(
        DTYPE
    )
    return result


def quadrature_solution(
    state: torch.Tensor, h: torch.Tensor, ell: torch.Tensor, time: float, grid: Grid
) -> torch.Tensor:
    """Evaluate all 64 independent unitary mixtures on the source grid."""
    result = torch.zeros_like(state, dtype=DTYPE)
    for point, weight in zip(grid.points, grid.kernel_weights, strict=True):
        result += weight * (
            torch.linalg.matrix_exp(-1j * time * (h + float(point) * ell))
            @ state.to(DTYPE)
        )
    return result
