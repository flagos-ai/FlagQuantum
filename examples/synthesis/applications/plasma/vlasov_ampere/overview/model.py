"""Fit the original degree-251 reciprocal for a freshly assembled plasma matrix."""

import math
from dataclasses import dataclass

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    evaluate,
    interpolation_grid,
    odd_table,
    relative_reciprocal,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
    fit,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.workflow import (
    shifted_phases,
)
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    physics,
    quantum,
)

DEGREE = 251


@dataclass(frozen=True)
class Model:
    matrix: torch.Tensor
    kappa: float
    normalization: float
    inverse_normalization: float
    coefficients: torch.Tensor
    fitted: PhaseFit
    phases: torch.Tensor
    source: torch.Tensor
    source_norm: float


def build(encoded: torch.Tensor | None = None) -> Model:
    matrix = physics.matrix() if encoded is None else encoded
    if matrix.shape != (128, 128) or not bool(torch.isfinite(matrix).all()):
        raise ValueError(
            "The original plasma operator requires a finite 128 by 128 matrix"
        )
    singular = torch.linalg.svdvals(matrix)
    kappa = 1 / float(singular[-1])
    cutoff = 1 / (kappa * quantum.NORMALIZATION)
    grid = interpolation_grid(DEGREE + 1)
    coefficients = (
        2
        / grid.numel()
        * (odd_table(grid, DEGREE).T @ relative_reciprocal(grid, cutoff, DEGREE))
    )
    samples = torch.linspace(0.0, 1.0, DEGREE * 25, dtype=torch.float64)
    maximum = float(evaluate(coefficients, samples).abs().max()) / math.cos(
        math.pi / 100
    )
    factor = 0.95 / maximum
    scaled = factor * coefficients
    fitted = fit(scaled)
    phases = shifted_phases(fitted)
    x = torch.linspace(0.0, 100.0, 8, dtype=torch.float64)
    current = 0.8 * torch.exp(-(x - 50).square() / (2 * 3**2))
    source_norm = float(current.norm())
    rhs = torch.zeros(128, dtype=torch.float64)
    rhs[64::8] = current / source_norm
    return Model(
        matrix,
        kappa,
        quantum.NORMALIZATION,
        kappa / factor,
        scaled,
        fitted,
        phases,
        rhs,
        source_norm,
    )
