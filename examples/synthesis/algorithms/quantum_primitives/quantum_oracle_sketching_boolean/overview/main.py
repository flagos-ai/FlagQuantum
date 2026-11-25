"""Run the original Boolean oracle sketches, scaling sweep, and QSVT correction."""

import argparse
from pathlib import Path

import numpy as np
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)

from .model import TRIALS
from .quantum import DEGREE, SHOTS, THRESHOLD, target_response
from .workflow import Result, run


def print_report(values: Result) -> None:
    bench = values.benchmark
    print("Boolean table:", list(map(int, bench.table)))
    print("Demo, Bell and unknown sample counts:", 200, 500, 20)
    print("Benchmark sizes and trials:", bench.sizes.size, TRIALS)
    print("Maximum benchmark size:", int(bench.sizes[-1]))
    print("M4000 spectral error:", bench.single_error)
    print(
        "Demo, known and Bell full-state errors:",
        values.demo_error,
        values.known_error,
        values.bell_error,
    )
    print("Shots per Bell circuit:", SHOTS)
    print(
        "Uncorrected exact and sampled Bell fidelity:",
        values.bell_exact,
        values.bell_sampled,
    )
    print(
        "Threshold, degree and phase count:",
        THRESHOLD,
        DEGREE,
        values.threshold.phases.numel(),
    )
    print(
        "Threshold approximation and phase fit errors:",
        values.threshold_error,
        values.threshold.residual,
    )
    filtered = values.corrected
    print(
        "Corrected physical bits and represented operations:",
        filtered.resources.n_qubits,
        filtered.resources.n_operations,
    )
    print(
        "Corrected oracle, spectral and full-state errors:",
        filtered.oracle_error,
        filtered.spectral_error,
        filtered.full_state_error,
    )
    print(
        "Corrected exact and sampled block probabilities:",
        filtered.success,
        filtered.sampled_success,
    )
    print(
        "Corrected exact and sampled conditional fidelity:",
        filtered.conditional_fidelity,
        filtered.sampled_fidelity,
    )


def save_figures(values: Result, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    bench = values.benchmark
    scaling = bench.table.size / bench.sizes
    single = float(np.median(bench.mean_error / np.sqrt(scaling)))
    mean = float(np.median(bench.mean_operator_error / scaling))
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.loglog(bench.sizes, bench.mean_error, "o-", label="Mean single-sketch error")
    axes.loglog(
        bench.sizes, bench.mean_operator_error, "s-", label="Mean operator error"
    )
    axes.loglog(bench.sizes, single * np.sqrt(scaling), "--", label="Square-root guide")
    axes.loglog(bench.sizes, mean * scaling, "--", label="Inverse-sample guide")
    axes.set(
        xlabel="Samples per trial",
        ylabel="Operator norm error",
        title="Boolean oracle sketching, 1,500 trials per size",
    )
    axes.legend()
    figure.savefig(directory / "oracle-scaling.png", dpi=150)
    plt.close(figure)

    grid = torch.linspace(0, 1, 1024, dtype=torch.float64)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot(grid, target_response(grid), label="Target threshold")
    axes.plot(
        grid,
        response(grid, values.threshold.phases),
        "--",
        label="Fitted QSVT response",
    )
    axes.axvline(THRESHOLD, color="gray", linestyle=":")
    axes.set(
        xlabel="Magnitude of the encoded sine",
        ylabel="Signed response",
        title="Unknown-distribution threshold",
    )
    axes.legend()
    figure.savefig(directory / "threshold-response.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("oracle-sketch-figures"))
    args = parser.parse_args()
    values = run()
    print_report(values)
    save_figures(values, args.figures)


if __name__ == "__main__":
    main()
