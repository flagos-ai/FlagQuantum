"""Regenerate the original 16-by-16 absorbing wave model and inverse polynomial."""

import cmath
import math
from dataclasses import dataclass

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
    fit,
    response,
)

GRID = 16
DATA_SIZE = GRID**2
DEGREE = 191
REAL = torch.float64
COMPLEX = torch.complex128
SOURCES = ((8, 2), (7, 2))


@dataclass(frozen=True)
class Model:
    matrix: torch.Tensor
    source: torch.Tensor
    reference: torch.Tensor
    omega: float
    diagonal: float
    alpha: float
    absorb_phase: float
    selector_amplitudes: tuple[float, ...]


@dataclass(frozen=True)
class Inverse:
    coefficients: torch.Tensor
    phases: PhaseFit
    minimum_sigma: float
    maximum_sigma: float
    expected_solution: torch.Tensor
    expected_fidelity: float


def build_model() -> Model:
    omega = 6 * math.pi / GRID
    diagonal = omega**2 - 4
    alpha = 4 + 3 * abs(diagonal)
    absorption = (1 + 0.5j * omega) / (-1 + 0.5j * omega)
    boundary_weight = torch.tensor(diagonal * absorption, dtype=COMPLEX)
    matrix = torch.zeros(DATA_SIZE, DATA_SIZE, dtype=COMPLEX)
    for x in range(GRID):
        for y in range(GRID):
            row = x * GRID + y
            boundary_x = x in (0, GRID - 1)
            boundary_y = y in (0, GRID - 1)
            if boundary_x and not boundary_y:
                matrix[row, (1 if x == 0 else GRID - 2) * GRID + y] = boundary_weight
            elif boundary_y and not boundary_x:
                matrix[row, x * GRID + (1 if y == 0 else GRID - 2)] = boundary_weight
            elif not boundary_x and not boundary_y:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    matrix[row, (x + dx) * GRID + y + dy] = 1
    barriers = [
        x * GRID + y for x in range(GRID) if (x >> 1) not in (2, 5) for y in (6, 7)
    ]
    matrix[barriers, :] = 0
    matrix[:, barriers] = 0
    matrix += diagonal * torch.eye(DATA_SIZE, dtype=COMPLEX)
    source = torch.zeros(DATA_SIZE, dtype=COMPLEX)
    for x, y in SOURCES:
        source[x * GRID + y] = 1 / math.sqrt(2)
    reference = torch.linalg.solve(matrix, source)
    probabilities = (4.0, 2 * abs(diagonal), abs(diagonal), 0.0)
    return Model(
        matrix,
        source,
        reference,
        omega,
        diagonal,
        alpha,
        cmath.phase(-absorption),
        tuple(math.sqrt(weight / alpha) for weight in probabilities),
    )


def fidelity(left: torch.Tensor, right: torch.Tensor) -> float:
    similarity = float(torch.vdot(left, right).abs() / (left.norm() * right.norm()))
    return similarity**2


def chebyshev(value: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate T on the real line, including the source mapping outside [-1, 1]."""
    previous, current = torch.ones_like(value), value
    for _ in range(1, degree):
        previous, current = current, 2 * value * current - previous
    return current


def inverse_fit(model: Model) -> Inverse:
    u, sigma, vh = torch.linalg.svd(model.matrix / model.alpha)
    minimum = float(sigma[-1])
    angles = math.pi * (torch.arange(DEGREE + 1, dtype=REAL) + 0.5) / (DEGREE + 1)
    x = angles.cos()
    offset = 1 + minimum**2
    scale = 1 - minimum**2
    degree = (DEGREE + 1) // 2
    signal = (2 * x.square() - offset) / scale
    denominator = chebyshev(torch.tensor(-offset / scale, dtype=REAL), degree)
    values = 0.75 * minimum * (1 - chebyshev(signal, degree) / denominator) / x
    orders = torch.arange(1, DEGREE + 1, 2, dtype=REAL)
    coefficients = (2 / angles.numel()) * torch.cos(angles[:, None] * orders).T @ values
    phases = fit(coefficients)
    expected = vh.mH @ (response(sigma, phases.phases) * (u.mH @ model.source))
    overlap = fidelity(expected, model.reference)
    if overlap < 0.95:
        raise ValueError("The original wave solution missed its SVD fidelity target")
    return Inverse(coefficients, phases, minimum, float(sigma[0]), expected, overlap)
