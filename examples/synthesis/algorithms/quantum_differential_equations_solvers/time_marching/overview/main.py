"""Run the full native time-marching experiment and save its plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)

from .approximation import REAL
from .workflow import SHOTS, Result, run


def print_report(result: Result) -> None:
    print(
        "Phase counts:",
        tuple(
            int(x.numel())
            for x in (
                result.exponential.even_fit.phases,
                result.exponential.odd_fit.phases,
                result.amplifier.phase_fit.phases,
            )
        ),
    )
    print("Scalar demonstration maximum error:", result.scalar_fit_error)
    print(
        "Amplifier interval error and maximum polynomial magnitude:",
        result.amplifier.interval_error,
        result.amplifier.maximum_magnitude,
    )
    print("Maximum Magnus block error:", result.magnus_block_error)
    print("Draws per circuit:", SHOTS)
    for label, march in (("Naive", result.naive), ("Amplified", result.amplified)):
        print(label + " counts:", march.accepted_counts)
        print(label + " success probability:", march.exact_prob)
        print(label + " normalized error:", march.normalized_error)
        print(label + " state error:", march.state_error)
        print(
            label + " resources:",
            march.resources.n_qubits,
            march.resources.n_operations,
            march.resources.depth,
        )
    print("Analytic final state:", result.classical[-1].tolist())


def save_figures(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for index in range(4):
        axes[0].plot(
            result.times.tolist(),
            result.classical[:, index].tolist(),
            label=f"Coordinate {index}",
        )
    axes[0].set(xlabel="Time", ylabel="Amplitude", title="Analytic diagonal evolution")
    axes[0].legend()
    exact = result.classical[-1] / result.classical[-1].norm()
    naive = result.naive.sampled_magnitudes / result.naive.sampled_magnitudes.norm()
    amplified = (
        result.amplified.sampled_magnitudes / result.amplified.sampled_magnitudes.norm()
    )
    bar_axes = torch.arange(4, dtype=REAL)
    for offset, values, label in (
        (-0.24, exact, "Analytic"),
        (0.0, naive, "Naive samples"),
        (0.24, amplified, "Amplified samples"),
    ):
        axes[1].bar(
            (bar_axes + offset).tolist(), values.tolist(), width=0.22, label=label
        )
    axes[1].set(
        xlabel="Coordinate",
        ylabel="Normalized amplitude",
        title="Counter-zero sample reconstruction",
    )
    axes[1].set_xticks(range(4))
    axes[1].legend()
    figure.savefig(directory / "time-marching.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    x = torch.linspace(-1, 1, 1000, dtype=REAL)
    demo = 0.5 * (
        response(x, result.demo.even_fit.phases)
        + response(x, result.demo.odd_fit.phases)
    )
    axes[0].plot(x.tolist(), demo.tolist(), label="Degree 7 and 6 responses")
    axes[0].plot(
        x.tolist(),
        (0.5 * (2 * (x - 1)).exp()).tolist(),
        "--",
        label="Scaled exponential",
    )
    axes[0].set(
        xlabel="Signal value",
        ylabel="Selected amplitude",
        title="Exponential approximation",
    )
    axes[0].legend()
    y = response(x, result.amplifier.phase_fit.phases)
    axes[1].plot(x.tolist(), y.tolist(), label="Bounded degree 7 response")
    interval = x[x.abs() <= 0.5]
    axes[1].plot(
        interval.tolist(), (1.8 * interval).tolist(), "--", label="Target interval"
    )
    axes[1].set(
        xlabel="Signal value",
        ylabel="Selected amplitude",
        title="Amplification approximation",
    )
    axes[1].legend()
    figure.savefig(directory / "time-polynomials.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("time-marching-figures"))
    arguments = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, arguments.figures)


if __name__ == "__main__":
    main()
