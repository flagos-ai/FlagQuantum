"""Draw the original graph, cost trace, sample histograms and selected sets."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .model import Model
from .workflow import SHOTS, Workflow


def draw_nodes(model: Model, destination: Path, title: str, word: int = 0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    positions = nx.spring_layout(model.graph, seed=8, method="force")
    selected = {vertex for vertex in model.graph if word & (1 << vertex)}
    fig, axes = plt.subplots(figsize=(7, 5))
    nx.draw_networkx(
        model.graph,
        positions,
        node_color=["tomato" if v in selected else "lightblue" for v in model.graph],
        edge_color=[
            "goldenrod" if a in selected or b in selected else "gray"
            for a, b in model.graph.edges
        ],
        ax=axes,
    )
    axes.set_title(title)
    axes.axis("off")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def draw_trace(result: Workflow, destination: Path) -> Path:
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot([row.cost for row in result.trace], color="royalblue")
    axes.set(
        xlabel="Actual objective evaluation",
        ylabel="Exact lower-tail CVaR (70%)",
        title="Six-layer QAOA: minimum dominating set",
    )
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def draw_histogram(result: Workflow, destination: Path) -> Path:
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
    fig.savefig(destination)
    plt.close(fig)
    return destination


def draw_results(result: Workflow, directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    best = result.optimized[0]
    return (
        draw_nodes(
            result.model, directory / "source-graph.png", "Original six-node graph"
        ),
        draw_trace(result, directory / "cost-trace.png"),
        draw_histogram(result, directory / "cost-histogram.png"),
        draw_nodes(
            result.model,
            directory / "observed-set.png",
            f"Best observed set: penalized cost {best.cost}, feasible={best.feasible}",
            best.decisions,
        ),
        draw_nodes(
            result.model,
            directory / "exact-set.png",
            "One exact minimum dominating set",
            result.classical[0],
        ),
    )
