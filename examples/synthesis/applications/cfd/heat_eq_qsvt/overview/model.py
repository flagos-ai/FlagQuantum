"""Rebuild the eight-position heat operator and fit its degree-three phases."""

import math
from dataclasses import dataclass

import torch

POINTS = 8
TIME_STEPS = 1024
DT = 1.0 / TIME_STEPS
DX = 1.0 / POINTS
OBSERVED_STEPS = 4
DTYPE = torch.float64


@dataclass(frozen=True)
class Model:
    positions: torch.Tensor
    times: torch.Tensor
    operator: torch.Tensor
    initial: torch.Tensor
    initial_norm: float
    scaling: float
    rotation: float
    kappa: float


def build_model() -> Model:
    positions = torch.linspace(0.0, 1.0, POINTS, dtype=DTYPE)
    times = torch.linspace(0.0, 2.0, TIME_STEPS, dtype=DTYPE)
    identity = torch.eye(POINTS, dtype=DTYPE)
    neighbor = torch.diag(torch.ones(POINTS - 1, dtype=DTYPE), diagonal=1)
    laplacian = 2 * identity - neighbor - neighbor.T
    operator = identity + (DT / DX**2) * laplacian
    initial = torch.sin(math.pi * positions)
    norm = float(initial.norm())
    ad = operator[1, :3]
    largest = float(ad.abs().max())
    return Model(
        positions,
        times,
        operator,
        initial / norm,
        norm,
        largest * ad.numel(),
        2 * math.acos(float(ad[0]) / largest),
        1 / float(torch.linalg.eigvalsh(operator)[0]),
    )


@dataclass(frozen=True)
class Fit:
    phases: tuple[float, ...]
    linear: float
    cubic: float
    max_error: float


def response(phases: tuple[float, ...], grid: torch.Tensor) -> torch.Tensor:
    """Evaluate the full Wx phase sequence in the source X measurement basis."""
    off = 1j * (1 - grid.square()).clamp_min(0).sqrt()
    w = torch.stack((grid, off, off, grid), dim=-1).reshape(-1, 2, 2)
    product = torch.eye(2, dtype=torch.complex128)
    signs = torch.tensor((1.0, -1.0), dtype=DTYPE)
    for index, angle in enumerate(phases):
        product = product @ torch.diag(torch.exp(1j * angle * signs))
        if index + 1 < len(phases):
            product = product @ w
    return 0.5 * product.sum(dim=(-2, -1))


def fit(model: Model) -> Fit:
    """Solve the phase convention of this degree-three Laurent instance."""
    degree_parameter = int(model.kappa**2 * math.log(model.kappa / 0.1))
    if degree_parameter != 2:
        raise ValueError("The bounded phase solver requires the source cubic")
    # The inverse approximation before rescaling is 2*x - x**3.
    peak_on_fit_interval = 2 * 0.8 - 0.8**3
    normalization = 0.9 / peak_on_fit_interval
    # Laurent completion shrinks the fit and adds eps/2 to the leading T3 term.
    first = 0.9999 * 1.25 * normalization
    third = 0.9999 * (0.00005 - 0.25 * normalization)
    linear, cubic = first - 3 * third, 4 * third
    peak_phase = -math.acos(first + third)

    # With phases (a, b, b, c), a = c - pi/2 removes the off-diagonal
    # contribution to the X readout. Matching the x^3 term then fixes b.
    low, high = math.pi / 6, math.pi / 4
    for _ in range(53):
        b = (low + high) / 2
        summed_ends = peak_phase - 2 * b
        trial = 4 * math.cos(summed_ends) * math.cos(b) ** 2
        if trial > cubic:
            low = b
        else:
            high = b
    middle = (low + high) / 2
    last = (peak_phase - 2 * middle + math.pi / 2) / 2
    phases = (last - math.pi / 2, middle, middle, last)
    grid = torch.linspace(-1, 1, 4097, dtype=DTYPE)
    target = linear * grid + cubic * grid**3
    error = float((response(phases, grid) - target).abs().max())
    if not error < 1e-12:
        raise ValueError("Fresh cubic phase synthesis failed the response check")
    return Fit(phases, linear, cubic, error)
