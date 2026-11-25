"""Fresh four-node thermal systems and reciprocal polynomial responses."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
    fit,
    response,
)

REAL = torch.float64
COMPLEX = torch.complex128
AMBIENT = 293
R_ENV = 0.001
HEAT = (2000.0, 4000.0, -200.0, -2000.0)
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
RESISTANCES = (0.005, 0.006, 0.006, 0.007, 0.007, 0.008)
CONFIGURATIONS = 64
TABLE = Path(__file__).parent / "data" / "costs.json"


@dataclass(frozen=True)
class Reciprocal:
    coefficients: torch.Tensor
    scale: float
    fit: PhaseFit
    minimum: float


def build_matrix(word: int = 63) -> torch.Tensor:
    """Regenerate one thermal matrix by its six conductance toggles."""
    if not 0 <= word < CONFIGURATIONS:
        raise ValueError("Expected a six-bit cooling configuration")
    matrix = torch.eye(4, dtype=REAL) / R_ENV
    for index, ((i, j), resistance) in enumerate(zip(EDGES, RESISTANCES, strict=True)):
        if (word >> (len(EDGES) - 1 - index)) & 1:
            value = 1 / resistance
            matrix[i, i] += value
            matrix[j, j] += value
            matrix[i, j] -= value
            matrix[j, i] -= value
    return matrix


def weights() -> tuple[float, ...]:
    return (*tuple(1 / value for value in RESISTANCES), 1 / (2 * R_ENV), 0.0)


def alpha() -> float:
    return 2 * math.fsum(weights())


def inverse_target(
    minimum: float, degree: int, *, target_scale: float | None = None
) -> Reciprocal:
    """Fit the odd mapped Chebyshev reciprocal at the source degree."""
    order = (degree + 1) // 2
    shift = 1 + minimum**2
    span = 1 - minimum**2
    x = (math.pi * (torch.arange(degree + 1, dtype=REAL) + 0.5) / (degree + 1)).cos()

    def recurrence(mapped: torch.Tensor) -> torch.Tensor:
        previous, value = torch.ones_like(mapped), mapped
        for _ in range(1, order):
            previous, value = value, 2 * mapped * value - previous
        return value

    normalizer = recurrence(torch.tensor(-shift / span, dtype=REAL))
    y = (1 - recurrence((2 * x.square() - shift) / span) / normalizer) / x
    degrees = torch.arange(1, degree + 1, 2, dtype=REAL)
    coefficients = (2 / x.numel()) * torch.cos(torch.acos(x)[:, None] * degrees).T @ y
    dense = torch.linspace(0, math.pi, 16385, dtype=REAL)
    max_p = float((torch.cos(dense[:, None] * degrees) @ coefficients).abs().max())
    amplitude_scale = 0.95 / max_p if target_scale is None else target_scale
    if amplitude_scale * max_p >= 1:
        raise ValueError("The scaled reciprocal must remain bounded")
    coefficients *= amplitude_scale
    return Reciprocal(coefficients, amplitude_scale, fit(coefficients), minimum)


def physical_solutions() -> torch.Tensor:
    """Regenerate all 64 classical candidate systems from their conductances."""
    matrices = torch.stack([build_matrix(word) for word in range(CONFIGURATIONS)])
    heat = torch.tensor(HEAT, dtype=REAL).expand(CONFIGURATIONS, -1)
    currents: torch.Tensor = torch.linalg.solve(matrices, heat)
    return currents


def load_demo_costs(path: Path = TABLE) -> torch.Tensor:
    """Read the source's separate one-phase-bit mock QAE experiment input."""
    loaded = json.loads(path.read_text())["costs"]
    tensor = torch.as_tensor(loaded, dtype=REAL)
    if tensor.shape != (CONFIGURATIONS,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Expected 64 finite mock-QAE data inputs")
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError("Demo costs must lie in [0,1]")
    return tensor


def spectral_solution(
    matrix: torch.Tensor, reciprocal: Reciprocal, rhs: torch.Tensor
) -> torch.Tensor:
    values, directions = torch.linalg.eigh(matrix / alpha())
    bounded = response(values, reciprocal.fit.phases)
    result: torch.Tensor = directions @ (bounded * (directions.T @ rhs))
    return result
