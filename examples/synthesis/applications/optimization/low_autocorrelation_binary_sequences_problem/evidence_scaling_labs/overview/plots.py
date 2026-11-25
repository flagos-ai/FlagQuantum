"""Draw the actual optimization queries and observed energy frequencies."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Result


def save(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6.6, 4.2))
    try:
        axes.plot(
            range(1, len(result.trace) + 1),
            [row.cost for row in result.trace],
            color="#326caa",
        )
        best = min(range(len(result.trace)), key=lambda i: result.trace[i].cost)
        axes.scatter(
            best + 1, result.trace[best].cost, color="#bc641e", label="Selected query"
        )
        axes.set_xlabel("Executed objective query")
        axes.set_ylabel("Mean source cost")
        axes.set_title("LABS optimization on the selected circuit")
        axes.legend()
        figure.tight_layout()
        figure.savefig(directory / "cost-convergence.png", dpi=140)
    finally:
        plt.close(figure)
    figure, axes = plt.subplots(figsize=(6.6, 4.2))
    try:
        total = sum(row.count for row in result.samples)
        axes.hist(
            [row.cost for row in result.samples],
            weights=[row.count / total for row in result.samples],
            bins=10,
            color="#326caa",
            edgecolor="#183444",
        )
        axes.set_xlabel("Source cost")
        axes.set_ylabel("Observed frequency")
        axes.set_title("LABS sampled costs")
        figure.tight_layout()
        figure.savefig(directory / "observed-costs.png", dpi=140)
    finally:
        plt.close(figure)
