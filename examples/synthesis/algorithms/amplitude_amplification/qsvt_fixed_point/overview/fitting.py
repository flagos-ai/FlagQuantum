"""Fit an odd Chebyshev polynomial and its physical projector rotations."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

DEGREE = 25
TARGET = 0.95
MINIMUM = 0.1
GRID_POINTS = 20_001
PHASE_SEED = 3
DTYPE = torch.float64
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class PhaseFit:
    """Fresh polynomial coefficients and a verified symmetric phase factorization."""

    coefficients: tuple[float, ...]
    phases: tuple[float, ...]
    target_error: float
    phase_error: float
    polynomial_iterations: int
    phase_evaluations: int
    amplitude_bound: float


def polynomial(points: RealArray, coefficients: tuple[float, ...]) -> RealArray:
    """Evaluate the odd Chebyshev series at real overlap values."""
    return np.asarray(np.polynomial.chebyshev.chebval(points, coefficients))


def _extrema(error: RealArray) -> list[int]:
    middle = np.flatnonzero(np.diff(error)[:-1] * np.diff(error)[1:] <= 0) + 1
    locations: list[int] = []
    for index in (0, *map(int, middle), len(error) - 1):
        if locations and error[locations[-1]] * error[index] > 0:
            if abs(error[index]) > abs(error[locations[-1]]):
                locations[-1] = index
        else:
            locations.append(index)
    return locations


def minimax_polynomial() -> tuple[tuple[float, ...], int]:
    """Fit degree 25 by exchanging alternating residual extrema."""
    design_count = (DEGREE + 1) // 2 + 1
    points = np.linspace(MINIMUM, 1.0, GRID_POINTS)
    nodes = (
        MINIMUM
        + (1 - MINIMUM) * (1 - np.cos(np.linspace(0, math.pi, design_count))) / 2
    )
    signs = (-1.0) ** np.arange(design_count)
    for iteration in range(1, 21):
        odd_terms = np.polynomial.chebyshev.chebvander(nodes, DEGREE)[:, 1::2]
        system = np.column_stack((odd_terms, signs))
        fit = np.linalg.solve(system, np.full(design_count, TARGET))
        coefficients = np.zeros(DEGREE + 1)
        coefficients[1::2] = fit[:-1]
        candidate = tuple(map(float, coefficients))
        extrema = _extrema(polynomial(points, candidate) - TARGET)
        if len(extrema) != design_count:
            raise ValueError("The polynomial fit lost an alternating extremum")
        following = points[extrema]
        if np.array_equal(following, nodes):
            return candidate, iteration
        nodes = following
    raise ValueError("Polynomial exchange did not stabilize within 20 iterations")


def phase_response(points: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """Read the aux-zero target amplitude from the two-projector subspace."""
    bad = torch.sqrt((1 - points.square()).clamp_min(0))
    source = torch.stack((points, bad), dim=1)
    amplitudes = source.to(torch.complex128)
    for index, angle in enumerate(phases):
        positive = torch.exp(0.5j * angle)
        negative = positive.conj()
        if index % 2 == 0:
            overlap = (amplitudes * source).sum(dim=1)
            amplitudes = (
                negative * amplitudes
                + (positive - negative) * overlap[:, None] * source
            )
        else:
            amplitudes = torch.stack(
                (positive * amplitudes[:, 0], negative * amplitudes[:, 1]), dim=1
            )
    return amplitudes[:, 0].real


def _mirrored(half: torch.Tensor) -> torch.Tensor:
    return torch.cat((half, half.flip(0)))


def _warm_phases() -> tuple[torch.Tensor, int]:
    """Find the desired phase branch before solving the collocation residual."""
    points = (
        MINIMUM
        + (1 - MINIMUM)
        * (1 - torch.cos(torch.linspace(0, math.pi, 129, dtype=DTYPE)))
        / 2
    )
    half = math.pi + 0.3 * torch.randn(
        13, dtype=DTYPE, generator=torch.Generator().manual_seed(PHASE_SEED)
    )
    half.requires_grad_(True)
    evaluations = 0
    for power in (2, 4, 8, 20, 40):
        optimizer: torch.optim.Optimizer = torch.optim.LBFGS(
            [half],
            lr=1,
            max_iter=180,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def loss_value(
            power: int = power, optimizer: torch.optim.Optimizer = optimizer
        ) -> float:
            nonlocal evaluations
            optimizer.zero_grad()
            error = (phase_response(points, _mirrored(half)) - TARGET).abs()
            if power == 2:
                loss = error.square().mean()
            else:
                loss = torch.logsumexp(power * (error + 1e-13).log(), dim=0) / power
            (half.grad,) = torch.autograd.grad(loss, half)
            evaluations += 1
            return float(loss.detach())

        optimizer.step(loss_value)
    return half.detach(), evaluations


def fit_source() -> PhaseFit:
    """Regenerate and independently verify the source polynomial and phases."""
    coefficients, iterations = minimax_polynomial()
    half, evaluations = _warm_phases()
    nodes = torch.cos(torch.arange(13, dtype=DTYPE) * math.pi / 26)
    reference = torch.from_numpy(polynomial(nodes.numpy(), coefficients))

    def residual(parameters: torch.Tensor) -> torch.Tensor:
        return phase_response(nodes, _mirrored(parameters)) - reference

    for _ in range(5):
        half.requires_grad_(True)
        error = residual(half)
        derivative = torch.stack(
            tuple(torch.autograd.grad(row, half, retain_graph=True)[0] for row in error)
        )
        half = half.detach() - torch.linalg.solve(derivative, error.detach())
    phases = _mirrored(half)
    grid = np.linspace(-1, 1, GRID_POINTS)
    expected = polynomial(grid, coefficients)
    actual = phase_response(torch.from_numpy(grid), phases).numpy()
    phase_error = float(np.max(np.abs(actual - expected)))
    amplitude_bound = float(np.max(np.abs(expected)))
    source_grid = np.linspace(MINIMUM, 1, GRID_POINTS)
    target_error = float(np.max(np.abs(polynomial(source_grid, coefficients) - TARGET)))
    if phase_error > 1e-10 or amplitude_bound > 1 or target_error > 0.035:
        raise ValueError(
            "The fitted polynomial or its phase factorization failed verification"
        )
    return PhaseFit(
        coefficients,
        tuple(map(float, phases)),
        target_error,
        phase_error,
        iterations,
        evaluations,
        amplitude_bound,
    )
