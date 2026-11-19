"""Regenerate the Lanchester system, Hermitian embedding and QPE eigenvalue grid."""

import math
from dataclasses import dataclass

import torch

DTYPE = torch.float64
COMPLEX = torch.complex128
N = 8
DT = 6
PHASE_WORDS = 64


def generate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regenerate the 16-equation model and its 32-dimensional Hermitian embedding."""
    operator = torch.eye(2 * N, dtype=DTYPE)
    for step in range(1, N):
        operator[step, step - 1] = -(1 - 0.01 * DT)
        operator[step, N + step - 1] = 0.1 * DT
        operator[N + step, N + step - 1] = -(1 - 0.02 * DT)
        operator[N + step, step - 1] = -0.02 * DT
    forcing = torch.cat(
        (
            torch.full((N,), 0.4 * DT, dtype=DTYPE),
            torch.full((N,), 0.01 * DT, dtype=DTYPE),
        )
    )
    forcing[0] = 30
    forcing[N] = 1
    zero = torch.zeros_like(operator)
    hermitian = torch.cat(
        (torch.cat((zero, operator.T), dim=1), torch.cat((operator, zero), dim=1)),
        dim=0,
    )
    padded = torch.cat((torch.zeros_like(forcing), forcing / forcing.norm()))
    return operator, forcing, hermitian, padded


@dataclass(frozen=True)
class Model:
    """Source system and all regenerated data required by native synthesis."""

    operator: torch.Tensor
    forcing: torch.Tensor
    hermitian: torch.Tensor
    padded: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    rescaled: torch.Tensor
    phase_eigenvalues: torch.Tensor
    evolution: torch.Tensor
    angles: torch.Tensor
    cutoff: float


def build_model() -> Model:
    """Shift and rescale the spectrum for six-bit phase estimation."""
    operator, forcing, hermitian, padded = generate()
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    span = float(eigenvalues[-1] - eigenvalues[0])
    lower = float(eigenvalues[0])
    scale = (1 - 1 / PHASE_WORDS) / span
    rescaled = (hermitian - lower * torch.eye(4 * N, dtype=DTYPE)) * scale
    phases = (eigenvalues - lower) * scale
    vectors = eigenvectors.to(COMPLEX)
    evolution = (vectors * torch.exp(2j * math.pi * phases)) @ vectors.mH
    cutoff = span / PHASE_WORDS
    grid = torch.arange(PHASE_WORDS, dtype=DTYPE) / PHASE_WORDS
    angles = 2 * torch.asin(torch.clamp(cutoff / (grid / scale + lower), -1, 1))
    return Model(
        operator,
        forcing,
        hermitian,
        padded,
        eigenvalues,
        eigenvectors,
        rescaled,
        phases,
        evolution,
        angles,
        cutoff,
    )
