"""Run the LiH phase-estimation workflow and write two diagnostic figures."""

import argparse
from pathlib import Path

from .workflow import EPS, POWERS, SEED, SHOTS, STEPS, Report, run_workflow


def print_report(result: Report) -> None:
    model = result.model
    print(
        "Tapered and retained terms:",
        len(model.coefficient_magnitudes),
        len(model.terms),
    )
    print("Normalization and offset:", model.normalization, model.offset)
    print("Full CI and active CI:", model.full_ci, model.active_ci)
    print("Coefficient difference:", model.reference_delta)
    print("Gamma and target error:", result.gamma, EPS)
    print("Powers:", POWERS)
    print("Adaptive bound:", result.adaptive_bound)
    print("Naive bound:", result.naive_bound)
    print("Experimental repetitions:", STEPS)
    print("QPE resources:", result.program.report["resources"])
    print("Shots and seed:", SHOTS, SEED)
    print("Most probable energy:", result.mode)
    print("Peaks above threshold:", result.peaks)
    print("Lowest sampled peak:", result.lowest_peak)
    print("Energy bin width:", model.normalization / len(result.phases))


def save_figures(result: Report, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.semilogy(
        range(1, len(result.model.coefficient_magnitudes) + 1),
        result.model.coefficient_magnitudes,
        ".",
        label="Fresh tapered terms",
    )
    axes.axhline(0.03, color="tab:red", linestyle="--", label="Truncation at 0.03 Ha")
    axes.set(
        xlabel="Term rank",
        ylabel="Coefficient magnitude (Ha)",
        title="LiH STO-3G Pauli coefficients",
    )
    axes.legend()
    figure.savefig(directory / "coefficient-magnitudes.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    visible = [
        (energy, count / SHOTS)
        for energy, count in zip(result.energies, result.counts, strict=True)
        if count
    ]
    x, y = zip(*visible, strict=True)
    axes.scatter(x, y, label=f"QPE observations ({SHOTS:,} draws)")
    axes.axvline(
        result.model.full_ci, color="tab:red", linestyle="--", label="Full CI energy"
    )
    axes.set(
        xlabel="Energy (Ha)",
        ylabel="Observed probability",
        title="LiH QPE energy histogram",
    )
    axes.legend()
    figure.savefig(directory / "energy-histogram.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("lih-qpe-figures"))
    args = parser.parse_args()
    result = run_workflow()
    print_report(result)
    save_figures(result, args.figures)


if __name__ == "__main__":
    main()
