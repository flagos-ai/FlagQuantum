"""Render observed optimization trace and measured cost distributions."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import SHOTS, Workflow


def draw_trace(result: Workflow, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(figsize=(8, 4))
    axes.plot([row.cost for row in result.trace])
    axes.set(
        xlabel="Actual objective evaluation",
        ylabel="Exact lower-tail CVaR (70%)",
        title="Original Set Cover: three QAOA layers",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def draw_histogram(result: Workflow, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(figsize=(8, 5))
    axes.hist(
        [[row.cost for row in result.optimized], [row.cost for row in result.uniform]],
        weights=[
            [row.count / SHOTS for row in result.optimized],
            [row.count / SHOTS for row in result.uniform],
        ],
        bins=50,
        alpha=0.7,
        label=["Optimized", "Uniform"],
    )
    axes.set(xlabel="Observed penalized cost", ylabel="Sample probability")
    axes.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
