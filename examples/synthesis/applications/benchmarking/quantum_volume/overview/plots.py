"""Separate original observed-count diagnostics from ideal heavy probabilities."""

from pathlib import Path

from matplotlib.figure import Figure

from .workflow import Experiment


def plot_experiment(experiment: Experiment, directory: Path) -> Figure:
    figure = Figure(figsize=(9, 5), constrained_layout=True)
    axes = figure.subplots()
    sizes = tuple(row.size for row in experiment.sizes)
    axes.errorbar(
        sizes,
        tuple(row.observed_heavy_probability for row in experiment.sizes),
        yerr=tuple(2 * row.source_standard_deviation for row in experiment.sizes),
        marker="o",
        capsize=4,
        label="Source observed-count score +/- 2 source deviations",
    )
    axes.plot(
        sizes,
        tuple(row.mean_ideal_heavy_probability for row in experiment.sizes),
        "s--",
        label="Ideal-probability heavy mass (separate diagnostic)",
    )
    axes.axhline(
        2 / 3, color="black", linestyle=":", label="Source decision threshold 2/3"
    )
    axes.set(
        xticks=sizes,
        xlabel="Qubits and layers",
        ylabel="Heavy-output probability / observed-count score",
        title="Original 40-circuit local experiment: 10 trials, 100 draws each",
        ylim=(0.6, 1.0),
    )
    axes.legend(loc="lower left", fontsize=9)
    axes.grid(alpha=0.2)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "observed_and_ideal_heavy_outputs.png", dpi=150)
    return figure
