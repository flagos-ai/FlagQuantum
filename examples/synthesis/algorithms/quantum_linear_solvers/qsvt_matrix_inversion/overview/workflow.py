"""Fit and execute the explicit and precision-driven matrix inversion routes."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    evaluate,
    interpolation_grid,
    odd_table,
    relative_reciprocal,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
    CTYPE,
    DTYPE,
    unitary,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
    fit,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.workflow import (
    build_circuit,
    independent_state,
    shifted_phases,
)

from .model import SIZE, Model, build_model

MANUAL_ERROR = 1e-4
AUTOMATIC_ERROR = 1e-2
SCALE = 0.25


@dataclass(frozen=True)
class Inverse:
    degree: int
    response_error: float
    phase_fit: PhaseFit
    solution: torch.Tensor
    full_state_error: float
    spectral_error: float
    solution_error: float
    projection_probability: float


def source_rhs() -> torch.Tensor:
    return torch.arange(SIZE, dtype=DTYPE)


def manual_degree(model: Model) -> int:
    degree = int(
        model.effective_condition * math.log(model.effective_condition / MANUAL_ERROR)
    )
    return degree + int(degree % 2 == 0)


def inverse_degree(model: Model, tolerance: float) -> int:
    """Bound the error of the unscaled inverse operator."""
    if not math.isfinite(tolerance) or not 0 < tolerance < SCALE:
        raise ValueError("Use a finite positive inverse tolerance below one quarter")
    a = 1 / model.effective_condition
    origin = (1 + a * a) / (1 - a * a)
    order = math.ceil(math.acosh(model.condition / tolerance) / math.acosh(origin))
    return 2 * order - 1


def inverse_coefficients(model: Model, degree: int) -> torch.Tensor:
    grid = interpolation_grid(degree + 1)
    target = SCALE * relative_reciprocal(grid, 1 / model.effective_condition, degree)
    return 2 / grid.numel() * (odd_table(grid, degree).T @ target)


def solve(model: Model, rhs: torch.Tensor, degree: int) -> Inverse:
    if rhs.shape != (SIZE,) or not bool(rhs.isfinite().all()) or float(rhs.norm()) == 0:
        raise ValueError("Provide four finite amplitudes with nonzero norm")
    if degree < 1 or degree > 255 or degree % 2 == 0:
        raise ValueError("Use an odd degree between one and 255")
    coefficients = inverse_coefficients(model, degree)
    a = 1 / model.effective_condition
    sigma = model.spectrum / model.encoding.alpha
    grid = torch.linspace(float(sigma[-1]), float(sigma[0]), 4097, dtype=DTYPE)
    polynomial_error = float(
        (evaluate(coefficients, grid) - SCALE * a / grid).abs().max()
    )
    phase_fit = fit(coefficients)
    shift = shifted_phases(phase_fit)
    block = unitary(model.encoding)
    circuit = build_circuit(model.encoding, block, shift, rhs)
    state = fq.run(circuit).statevector().flatten()
    full_reference = independent_state(block, shift, rhs)
    torch.testing.assert_close(state, full_reference, atol=1e-10, rtol=0)
    sign = (-1) ** ((degree - 1) // 2)
    scale = sign * rhs.norm() * model.condition / SCALE
    solution = scale * state[:SIZE]
    u, singular, vh = torch.linalg.svd(model.matrix)
    spectral = vh.mH @ (
        evaluate(coefficients, singular / model.encoding.alpha) * (u.mH @ rhs.to(CTYPE))
    )
    spectral *= rhs.new_tensor(model.condition / SCALE)
    torch.testing.assert_close(solution, spectral, atol=1e-10, rtol=0)
    expected = torch.linalg.solve(model.matrix, rhs.to(CTYPE))
    return Inverse(
        degree,
        polynomial_error,
        phase_fit,
        solution.real,
        float((state - full_reference).abs().max()),
        float((solution - spectral).abs().max()),
        float((solution - expected).norm()),
        float(state[:SIZE].abs().square().sum()),
    )


def run_workflow() -> tuple[Model, Inverse, Inverse]:
    model = build_model()
    b = source_rhs()
    return (
        model,
        solve(model, b, manual_degree(model)),
        solve(model, b, inverse_degree(model, AUTOMATIC_ERROR)),
    )
