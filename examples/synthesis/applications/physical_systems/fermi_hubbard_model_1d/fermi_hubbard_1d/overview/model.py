"""Fresh source Hamiltonians, Slater factors and independent determinant amplitudes."""

import itertools
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SlaterPlan:
    rotations: tuple[tuple[int, float], ...]
    negative_pivots: tuple[int, ...]


def example_matrix() -> torch.Tensor:
    return torch.tensor(
        [
            [0.8, 0.25, 0, 0.15],
            [0.25, -0.4, -0.35, 0],
            [0, -0.35, 0.3, 0.2],
            [0.15, 0, 0.2, -0.1],
        ],
        dtype=torch.float64,
    )


def hamiltonian(strength: float, sigma: float, hopping: float = 1.0) -> torch.Tensor:
    matrix = torch.zeros((8, 8), dtype=torch.float64)
    for site in range(4):
        matrix[site, site] = -strength * math.exp(-((site - 1.5) ** 2) / (2 * sigma**2))
    for spin in range(2):
        for site in range(3):
            a, b = site + 4 * spin, site + 4 * spin + 1
            matrix[a, b] = matrix[b, a] = -hopping
    return matrix


def factor(orbitals: torch.Tensor) -> SlaterPlan:
    if orbitals.ndim != 2 or orbitals.dtype != torch.float64:
        raise ValueError("orbitals must be a real rank-two float64 tensor")
    rows, occupied = orbitals.shape
    if (
        rows not in (4, 8)
        or not 1 <= occupied <= rows
        or not torch.isfinite(orbitals).all()
    ):
        raise ValueError("expect finite columns for four or eight modes")
    torch.testing.assert_close(
        orbitals.T @ orbitals,
        torch.eye(occupied, dtype=torch.float64),
        atol=3e-14,
        rtol=0,
    )
    working = orbitals.clone()
    rotations = []
    for column in range(occupied):
        for right in range(rows - 1, column, -1):
            left = right - 1
            a, b = (float(working[i, column]) for i in (left, right))
            if abs(b) < 1e-15:
                continue
            theta = math.atan2(b, a)
            c, s = math.cos(theta), math.sin(theta)
            top, bottom = working[left].clone(), working[right].clone()
            working[left], working[right] = c * top + s * bottom, -s * top + c * bottom
            rotations.append((left, theta))
    signs = tuple(int(float(working[i, i]) < 0) for i in range(occupied))
    target = torch.zeros_like(working)
    for i, negative in enumerate(signs):
        target[i, i] = (-1) ** negative
    torch.testing.assert_close(working, target, rtol=0, atol=3e-14)
    return SlaterPlan(tuple(reversed(rotations)), signs)


def expected_state(orbitals: torch.Tensor) -> torch.Tensor:
    rows, occupied = orbitals.shape
    result = torch.zeros(1 << rows, dtype=torch.complex128)
    for subset in itertools.combinations(range(rows), occupied):
        word = sum(1 << wire for wire in subset)
        result[word] = torch.linalg.det(orbitals[list(subset), :])
    return result


def reconstruct_orbitals(plan: SlaterPlan, modes: int) -> torch.Tensor:
    """Apply inverse row eliminations to the signed reference block."""
    occupied = len(plan.negative_pivots)
    matrix = torch.zeros((modes, occupied), dtype=torch.float64)
    for index, negative in enumerate(plan.negative_pivots):
        matrix[index, index] = (-1) ** negative
    for left, theta in plan.rotations:
        c, s = math.cos(theta), math.sin(theta)
        first, second = matrix[left].clone(), matrix[left + 1].clone()
        matrix[left], matrix[left + 1] = c * first - s * second, s * first + c * second
    return matrix
