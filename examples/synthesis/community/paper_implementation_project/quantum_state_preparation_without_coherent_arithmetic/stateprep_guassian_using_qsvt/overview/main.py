"""Prepare the original 256-point Gaussian state with new QSVT phases."""

import argparse
from pathlib import Path

import torch

from .workflow import SIZE, WIDTH, Result, run


def print_report(result: Result) -> None:
    print("Data bits and Gaussian points:", WIDTH, SIZE)
    print(
        "Even and odd phase counts:",
        result.even.phases.numel(),
        result.odd.phases.numel(),
    )
    print("Even and odd polynomial errors:", *result.fit_error)
    print("Even and odd phase fit errors:", result.even.residual, result.odd.residual)
    print(
        "Continuum and measured success amplitudes:",
        result.continuum_overlap,
        result.measured_overlap,
    )
    print(
        "Independent preparation and amplification errors:",
        result.prep_error,
        result.amplified_error,
    )
    print(
        "Zero-block probability and workspace leakage:",
        result.accepted_mass,
        result.work_probability,
    )
    print("Probability MSE:", result.mse)
    print(
        "Native physical bits, operations and depth:",
        result.resources.n_qubits,
        result.resources.n_operations,
        result.resources.depth,
    )
    print(
        "CX/U3 physical bits, depth and CX count:",
        result.target_resources.n_qubits,
        result.target_resources.depth,
        result.target_resources.operation_counts["cx"],
    )


def save_figures(result: Result, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    coordinates = 4 * torch.arange(SIZE, dtype=torch.float64) / SIZE - 2
    fig, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot(coordinates, result.expected_distribution, label="Discrete Gaussian")
    axes.scatter(coordinates, result.distribution, s=10, label="QSVT and amplification")
    axes.set(
        xlabel="Gaussian coordinate",
        ylabel="Probability",
        title="Eight-bit Gaussian preparation",
    )
    axes.legend()
    fig.savefig(directory / "gaussian-probability.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot(coordinates, result.distribution - result.expected_distribution)
    axes.axhline(0, color="gray", linestyle=":")
    axes.set(
        xlabel="Gaussian coordinate",
        ylabel="Observed minus target probability",
        title="Gaussian preparation residual",
    )
    fig.savefig(directory / "gaussian-residual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("gaussian-qsvt-figures"))
    args = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, args.figures)


if __name__ == "__main__":
    main()
