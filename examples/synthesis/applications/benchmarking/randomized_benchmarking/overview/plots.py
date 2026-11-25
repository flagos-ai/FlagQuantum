"""Actual survival observations, with no identifiable noisy-device decay claim."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import MODES, Experiment


def plot_experiment(experiment: Experiment, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4))
    for index, mode in enumerate(MODES):
        trials = sorted(
            (row for row in experiment.trials if row.mode == mode),
            key=lambda row: row.length,
        )
        axes.plot(
            [row.length for row in trials],
            [row.survival for row in trials],
            "o-" if index == 0 else "x--",
            label=f"Local {mode.replace('_', ' ')}",
        )
    axes.set(
        xlabel="Random Clifford word length",
        ylabel="Observed survival probability",
        title="Ideal local sequence/inverse experiment",
        ylim=(0.95, 1.01),
    )
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path = directory / "survival.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
