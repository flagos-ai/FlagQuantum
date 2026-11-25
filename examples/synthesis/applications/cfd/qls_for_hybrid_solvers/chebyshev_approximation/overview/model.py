"""Build the nozzle singular-value interval and four reciprocal approximations."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import torch

DTYPE = torch.float64
SCALE = 0.5
EPSILON = 1e-4
PREPARATION_ERROR = 0.03
DegreeMethod = Literal["relative", "uniform", "cks"]
APPROXIMATIONS: tuple[tuple[int, DegreeMethod], ...] = (
    (127, "relative"),
    (255, "relative"),
    (255, "uniform"),
    (255, "cks"),
)


class NozzleCsr(TypedDict):
    shape: list[int]
    indptr: list[int]
    indices: list[int]
    data: list[float]


@dataclass(frozen=True)
class Model:
    alpha: float
    smallest: float
    largest: float
    cks_power: int
    cks_degree: int


@dataclass(frozen=True)
class Approximation:
    degree: int
    method: DegreeMethod
    coefficients: torch.Tensor
    normalization: float
    relative_error: float
    uniform_error: float


def source_matrix() -> torch.Tensor:
    """Restore the original 16 by 16 CSR matrix without a sparse-data runtime."""
    values: NozzleCsr = json.loads(
        Path(__file__).with_name("nozzle_008_csr.json").read_text()
    )
    if values["shape"] != [16, 16] or len(values["data"]) != 45:
        raise ValueError("Expected the original sixteen-row nozzle matrix")
    result = torch.zeros((16, 16), dtype=DTYPE)
    for row, (start, end) in enumerate(
        zip(values["indptr"][:-1], values["indptr"][1:], strict=True)
    ):
        result[row, values["indices"][start:end]] = torch.tensor(
            values["data"][start:end], dtype=DTYPE
        )
    return result


def build_model() -> Model:
    matrix = source_matrix()
    alpha = math.fsum(
        float(matrix.diagonal(offset).abs().max())
        for offset in range(1 - matrix.size(0), matrix.size(1))
    )
    spectrum = torch.linalg.svdvals(matrix / alpha)
    low, high = float(spectrum[-1]), float(spectrum[0])
    kappa = 1 / low
    power = int(kappa**2 * math.log(kappa / EPSILON))
    degree = 2 * int(math.sqrt(power * math.log(4 * power / EPSILON))) + 1
    return Model(alpha, low, high, power, degree)


def odd_table(x: torch.Tensor, degree: int) -> torch.Tensor:
    angles = torch.arccos(x.clamp(-1, 1))
    orders = torch.arange(1, degree + 1, 2, dtype=DTYPE)
    return torch.cos(angles[..., None] * orders)


def evaluate(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return odd_table(x, 2 * coefficients.numel() - 1) @ coefficients


def interpolation_grid(points: int) -> torch.Tensor:
    return torch.cos((torch.arange(points, dtype=DTYPE) + 0.5) * (math.pi / points))


def chebyshev_series(x: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    angles = torch.arccos(x.clamp(-1, 1))
    outside = torch.arccosh((-x).clamp(min=1))
    return torch.where(
        x[..., None] >= -1,
        torch.cos(angles[..., None] * orders),
        (1 - 2 * (orders % 2)) * torch.cosh(outside[..., None] * orders),
    )


def relative_reciprocal(x: torch.Tensor, cutoff: float, degree: int) -> torch.Tensor:
    """Evaluate the relative minimax residual on the squared-value interval."""
    order = torch.tensor([(degree + 1) // 2], dtype=DTYPE)
    origin = -(1 + cutoff**2) / (1 - cutoff**2)
    coordinate = origin + 2 * x.square() / (1 - cutoff**2)
    t0 = chebyshev_series(torch.tensor(origin, dtype=DTYPE), order)[0]
    residual = chebyshev_series(coordinate, order)[..., 0] / t0
    return cutoff * (1 - residual) / x


def uniform_basis(x: torch.Tensor, low: float, orders: torch.Tensor) -> torch.Tensor:
    coordinate = (2 * x.square() - 1 - low**2) / (1 - low**2)
    return x[..., None] * chebyshev_series(coordinate, orders)


def uniform_reciprocal(grid: torch.Tensor, cutoff: float, degree: int) -> torch.Tensor:
    """Minimize the absolute residual by bounded Lawson reweighting."""
    phases = torch.linspace(0, math.pi, 4097, dtype=DTYPE)
    x = ((1 + cutoff**2 + (1 - cutoff**2) * torch.cos(phases)) / 2).sqrt()
    orders = torch.arange((degree + 1) // 2, dtype=DTYPE)
    basis = uniform_basis(x, cutoff, orders)
    target = cutoff / x
    weights = torch.ones_like(x)
    best = torch.zeros_like(orders)
    best_error = math.inf
    for _ in range(160):
        weight_root = weights.sqrt()
        solution = torch.linalg.lstsq(
            basis * weight_root[:, None], target * weight_root, driver="gels"
        ).solution
        error = (basis @ solution - target).abs()
        worst = float(error.max())
        if worst < best_error:
            best, best_error = solution, worst
        weights *= (error / worst).clamp(min=1e-3).pow(0.65)
        weights /= weights.max()
        weights.clamp_(min=1e-10)
    return uniform_basis(grid, cutoff, orders) @ best


def coefficient_approximation(
    nozzle: Model, degree: int, method: DegreeMethod
) -> Approximation:
    """Recover odd Chebyshev coefficients and measure error on the source interval."""
    if (degree, method) not in APPROXIMATIONS:
        raise ValueError("Use one of the four original approximation configurations")
    x = interpolation_grid(degree + 1)
    if method == "relative":
        y = relative_reciprocal(x, nozzle.smallest, degree)
    elif method == "uniform":
        y = uniform_reciprocal(x, nozzle.smallest, degree)
    else:
        x = interpolation_grid(nozzle.cks_degree + 1)
        y = (
            -SCALE
            * nozzle.smallest
            * torch.expm1(nozzle.cks_power * torch.log1p(-x.square()))
            / x
        )
    coeffs = (2 / x.numel()) * (odd_table(x, degree).T @ y)
    if method == "cks":
        normalization = SCALE * nozzle.smallest
    else:
        # Measure the polynomial's global peak on a dense Gauss-Chebyshev grid.
        # This matches the workflow's scaling step without a hosted optimizer.
        coeffs = coeffs / nozzle.smallest
        peak = float(evaluate(coeffs, interpolation_grid(32768)).abs().max())
        normalization = SCALE / peak
        coeffs = normalization * coeffs
    values = torch.linspace(nozzle.smallest, nozzle.largest, 2000, dtype=DTYPE)
    scale = SCALE * nozzle.smallest / normalization
    estimate = scale * evaluate(coeffs, values)
    target = SCALE * nozzle.smallest / values
    return Approximation(
        degree,
        method,
        coeffs,
        normalization,
        float((estimate / target - 1).abs().max()),
        float((estimate - target).abs().max()),
    )


def fit_alternating(coefficients: torch.Tensor) -> torch.Tensor:
    """Fit the source line through the first and last even-index magnitudes."""
    magnitudes = coefficients.abs()
    slope = (magnitudes[-2] - magnitudes[0]) / (magnitudes.numel() - 2)
    values = torch.arange(magnitudes.numel(), dtype=DTYPE)
    return (1 - 2 * (values % 2)) * (magnitudes[0] + slope * values)
