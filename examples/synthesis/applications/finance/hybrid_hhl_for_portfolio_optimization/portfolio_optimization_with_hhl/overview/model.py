"""Regenerate the original two-asset matrix, spectrum and inverse table."""

import math
from dataclasses import dataclass

import torch

DTYPE = torch.float64
COMPLEX = torch.complex128
PHASE_BITS = 4
REDUCED_BITS = 3
SHOTS = 2048


@dataclass(frozen=True)
class Model:
    operator: torch.Tensor
    normalized_input: torch.Tensor
    direct: torch.Tensor
    comparison: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalue_bound: float
    scale: float
    cutoff: float
    evolution: torch.Tensor
    basic_angles: tuple[float, ...]


def build_model() -> Model:
    covariance = torch.tensor(((0.2, 0.04), (0.04, 0.4)), dtype=DTYPE)
    constraints = torch.tensor(((0.04, 0.64), (1.0, 0.8)), dtype=DTYPE)
    operator = torch.cat(
        (
            torch.cat((torch.zeros((2, 2), dtype=DTYPE), constraints), dim=1),
            torch.cat((constraints.T, covariance), dim=1),
        )
    )
    forcing = torch.tensor((2.4, 1.0, 0.0, 0.0), dtype=DTYPE)
    normalized = forcing / forcing.norm()
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    bound = float(eigenvalues.abs().max())
    resolution = 1 << PHASE_BITS
    scale = (1 - 1 / resolution) / (2 * bound)
    cutoff = bound / (resolution - 1)
    shifted = (operator + bound * torch.eye(4, dtype=DTYPE)) * scale
    evolution = torch.linalg.matrix_exp(shifted.to(COMPLEX) * (2j * math.pi))
    bins = torch.arange(resolution, dtype=DTYPE) / resolution
    angles = 2 * torch.asin(torch.clamp(cutoff / (bins / scale - bound), -1, 1))
    direct = torch.linalg.solve(operator, normalized)
    return Model(
        operator,
        normalized,
        direct,
        direct / direct.norm(),
        eigenvalues,
        eigenvectors,
        bound,
        scale,
        cutoff,
        evolution,
        tuple(map(float, angles)),
    )


def reduced_table(model: Model, words: tuple[int, ...]) -> tuple[float, ...]:
    """Accumulate the original selected rotations into each reduced raw word."""
    angles = [0.0] * (1 << REDUCED_BITS)
    for word in words:
        phase = word / (1 << PHASE_BITS)
        eigenvalue = phase / model.scale - model.eigenvalue_bound
        angle = 2 * math.asin(model.cutoff / eigenvalue)
        angles[word >> (PHASE_BITS - REDUCED_BITS)] += angle
    return tuple(angles)
