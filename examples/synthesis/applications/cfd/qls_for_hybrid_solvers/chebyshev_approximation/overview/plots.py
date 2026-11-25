"""Plot the four approximations and the measured coefficient preparation."""

from pathlib import Path

import torch

from .model import DTYPE, SCALE, evaluate
from .workflow import Result

NAMES = {
    "relative": "Relative-error minimax",
    "uniform": "Uniform-error fit",
    "cks": "Truncated CKS",
}


def save_figures(result: Result, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    nozzle = result.nozzle
    x = torch.linspace(0, nozzle.largest, 500, dtype=DTYPE)
    mask = x >= nozzle.smallest
    reciprocal = SCALE * nozzle.smallest / x[mask]

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for ax, case in zip(axs.flat, result.approximations, strict=True):
        scale = SCALE * nozzle.smallest / case.normalization
        ax.plot(x[mask], reciprocal, color="black", label="Scaled reciprocal")
        ax.plot(x, scale * evaluate(case.coefficients, x), "--", label="Approximation")
        ax.axvline(nozzle.smallest, linestyle=":", color="gray")
        ax.set(
            xlabel="Normalized singular value",
            ylabel="Polynomial value",
            title=f"{NAMES[case.method]}, degree {case.degree}",
        )
        ax.legend(loc="upper right")
        ax.text(
            0.98,
            0.12,
            f"Max relative error {case.relative_error:.3f}\n"
            f"Max absolute error {case.uniform_error:.4f}",
            transform=ax.transAxes,
            horizontalalignment="right",
        )
    fig.savefig(directory / "reciprocal-approximations.png", dpi=150)
    plt.close(fig)

    original = result.approximations[1]
    scale = SCALE * nozzle.smallest / original.normalization
    fig, (coeffs, function) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for label, values in (
        ("Original degree 255", original.coefficients),
        ("Exact fitted preparation", result.exact.coefficients),
        ("Bounded approximate preparation", result.approximate.coefficients),
    ):
        coeffs.plot(range(values.numel()), values.abs(), label=label)
        function.plot(x, scale * evaluate(values, x), label=label)
    coeffs.set(
        xlabel="Odd coefficient index",
        ylabel="Coefficient magnitude",
        title="LCU weights",
    )
    coeffs.legend()
    function.plot(x[mask], reciprocal, "--", color="black", label="Scaled reciprocal")
    function.axvline(nozzle.smallest, linestyle=":", color="gray")
    function.set(
        xlabel="Normalized singular value",
        ylabel="Polynomial value",
        title="Resulting polynomials",
    )
    function.legend()
    fig.savefig(directory / "prepared-coefficients.png", dpi=150)
    plt.close(fig)
