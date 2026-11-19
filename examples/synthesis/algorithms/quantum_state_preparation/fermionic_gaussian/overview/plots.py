"""Plot the actual orbital amplitudes in source mode order."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Slater


def plot_slater(slater: Slater, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlap = (slater.orbital.conj() * slater.amplitudes).sum()
    observed = slater.amplitudes / (overlap / overlap.abs())
    modes = range(8)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.bar(
        [mode - 0.17 for mode in modes],
        slater.orbital.real,
        width=0.34,
        label="Exact orbital",
    )
    axes.bar(
        [mode + 0.17 for mode in modes],
        observed.real,
        width=0.34,
        label="Native (phase aligned)",
    )
    axes.set(
        title="One electron on the four-site ring",
        xlabel="Mode",
        ylabel="Real amplitude",
        xticks=modes,
    )
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
