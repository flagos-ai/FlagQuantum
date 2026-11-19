"""Approximate the original t=22 cosine/sine with native Torch Chebyshev math."""

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

EVOLUTION_TIME = 22.0
EPS = 1e-7


def degree_for_error(time: float, error: float) -> int:
    """Bound the exact Jacobi-Anger tail on the Bernstein ellipse rho=2."""
    if not math.isfinite(time) or not math.isfinite(error) or not 0 < error < 1:
        raise ValueError("Use finite time and approximation error in (0, 1).")
    return math.ceil((math.log(2 / error) + 0.75 * abs(time)) / math.log(2))


def coefficients(degree: int, time: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Project e^(itx) on Chebyshev modes by 512-node midpoint cosine quadrature."""
    if (
        type(degree) is not int
        or not 0 <= degree <= 128
        or not math.isfinite(time)
        or abs(time) > 32
    ):
        raise ValueError("This local example bounds degree <= 128 and |time| <= 32.")
    theta = (torch.arange(512, dtype=torch.float64) + 0.5) * math.pi / 512
    modes = torch.arange(degree + 1, dtype=torch.float64)
    basis = torch.cos(modes[:, None] * theta[None, :])
    values = torch.exp(1j * time * torch.cos(theta))
    coeff = (2 / 512) * (basis.to(torch.complex128) @ values)
    coeff[0] /= 2
    even = (torch.arange(degree + 1) % 2) == 0
    cosine = torch.where(even, coeff.real, 0.0)
    sine = torch.where(~even, coeff.imag, 0.0)
    return cosine, sine


def evaluate(coefficients: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Evaluate Chebyshev coefficients by the Clenshaw recurrence."""
    previous = torch.zeros_like(xs)
    following = torch.zeros_like(xs)
    for coefficient in reversed(coefficients[1:]):
        current = coefficient + 2 * xs * previous - following
        following, previous = previous, current
    return coefficients[0] + xs * previous - following


def run_workflow() -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    degree = degree_for_error(EVOLUTION_TIME, EPS)
    cosine, sine = coefficients(degree, EVOLUTION_TIME)
    xs = torch.linspace(0, 1, 100, dtype=torch.float64)
    cos_approx = evaluate(cosine, xs)
    sin_approx = evaluate(sine, xs)
    assert float((cos_approx - torch.cos(EVOLUTION_TIME * xs)).abs().max()) < EPS
    assert float((sin_approx - torch.sin(EVOLUTION_TIME * xs)).abs().max()) < EPS
    print("Polynomial degree:", degree)
    return degree, xs, cos_approx, sin_approx


def plot_results(
    xs: torch.Tensor, cos_approx: torch.Tensor, sin_approx: torch.Tensor
) -> "Figure":
    import matplotlib.pyplot as plt

    exact_cos = torch.cos(EVOLUTION_TIME * xs)
    exact_sin = torch.sin(EVOLUTION_TIME * xs)
    figure, (approximation, error) = plt.subplots(1, 2, figsize=(14, 5))
    approximation.plot(xs, exact_cos, "-r", linewidth=3, label="cos")
    approximation.plot(xs, cos_approx, "--k", linewidth=2, label="approx. cos")
    approximation.plot(xs, exact_sin, "-y", linewidth=3, label="sin")
    approximation.plot(xs, sin_approx, "--b", linewidth=2, label="approx. sin")
    approximation.set_ylabel("cos(t*x), sin(t*x)", fontsize=16)
    approximation.set_title("Approximation", fontsize=16)
    error.plot(xs, exact_cos - cos_approx, "-r", linewidth=3, label="cos error")
    error.plot(xs, exact_sin - sin_approx, "-y", linewidth=3, label="sin error")
    error.set_ylabel("Error", fontsize=16)
    error.set_title("Approximation Error", fontsize=16)
    for axes in (approximation, error):
        axes.set_xlabel("x", fontsize=16)
        axes.tick_params(labelsize=16)
        axes.legend(loc="lower left", fontsize=14)
        axes.set_xlim(-0.1, 1.1)
    figure.tight_layout()
    plt.show()
    return figure


def run_example() -> None:
    _, xs, cosine, sine = run_workflow()
    plot_results(xs, cosine, sine)


if __name__ == "__main__":
    run_example()
