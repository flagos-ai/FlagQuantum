"""Run the original QSVT Hamiltonian evolution and compare complex amplitudes."""

import argparse
from pathlib import Path

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)

from .workflow import ALPHA, DEFAULT_SEED, DEGREE, EPS, SCALE, TIME, Result, run


def print_report(result: Result) -> None:
    print("Time and error target:", TIME, EPS)
    print("Data bits, Hamiltonian block bits and scale:", 2, 2, ALPHA)
    print(
        "Polynomial degree and cosine and sine phase counts:",
        DEGREE,
        result.cosine.phases.numel(),
        result.sine.phases.numel(),
    )
    print(
        "Cosine and sine phase fit errors:",
        result.cosine.residual,
        result.sine.residual,
    )
    print(
        "Cosine and sine approximation errors:", result.cosine_error, result.sine_error
    )
    print("Block and full-state errors:", result.block_error, result.full_state_error)
    print("Evolution L2 error:", result.evolution_error)
    print(
        "Overlap and projection probability:",
        result.overlap,
        result.success_probability,
    )
    print(
        "Physical bits and represented operations:",
        result.resources.n_qubits,
        result.resources.n_operations,
    )
    print("Expected complex data state:", tuple(map(complex, result.expected)))
    print("Observed complex data state:", tuple(map(complex, result.state)))
    print("Phase fit seconds:", result.phase_seconds)


def save_figures(result: Result, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    grid = torch.linspace(-1, 1, 501, dtype=torch.float64)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot(
        grid,
        response(grid, result.cosine.phases) - SCALE * (TIME * ALPHA * grid).cos(),
        label="Cosine",
    )
    axes.plot(
        grid,
        response(grid, result.sine.phases) - SCALE * (TIME * ALPHA * grid).sin(),
        label="Sine",
    )
    axes.set(
        xlabel="Normalized eigenvalue",
        ylabel="Response error",
        title="Jacobi–Anger truncation and fitted response",
    )
    axes.legend()
    figure.savefig(directory / "qsvt-response-errors.png", dpi=150)
    plt.close(figure)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, expected, actual, title in (
        (left, result.expected.real, result.state.real, "Real amplitude"),
        (right, result.expected.imag, result.state.imag, "Imaginary amplitude"),
    ):
        ax.plot(range(4), expected, "o", label="Matrix exponential")
        ax.plot(range(4), actual, "x", label="QSVT")
        ax.set(xticks=list(range(4)), xlabel="Basis index", ylabel=title)
        ax.legend()
    figure.suptitle("Two-bit Hamiltonian evolution")
    figure.savefig(directory / "qsvt-evolution-state.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("qsvt-evolution-figures"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = run(args.seed)
    print_report(result)
    save_figures(result, args.figures)


if __name__ == "__main__":
    main()
