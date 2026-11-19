"""Plot actual optimizer queries and the direct and variational distributions."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Solve


def save_trace(result: Solve, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.plot(range(1, len(result.queries) + 1), [row.cost for row in result.queries])
    axes.set(
        title="VQLS sampled objective",
        xlabel="Actual objective query",
        ylabel="Conditional cost",
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def save_comparison(result: Solve, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axes, title, state in (
        (left, "Direct linear solve", result.reference),
        (right, "Variational state", result.amplitudes),
    ):
        axes.bar(range(8), state.abs().square())
        axes.set(title=title, xlabel="Data word", xticks=range(8))
    left.set_ylabel("Probability")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
