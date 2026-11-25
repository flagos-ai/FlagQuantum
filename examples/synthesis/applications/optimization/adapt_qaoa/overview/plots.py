"""Render measured objective evaluation traces."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Optimization


def plot_cost(result: Optimization, title: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.plot(range(1, len(result.trace) + 1), [row.cost for row in result.trace])
    axes.set(title=title, xlabel="Objective evaluation", ylabel="Exact mean energy")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
