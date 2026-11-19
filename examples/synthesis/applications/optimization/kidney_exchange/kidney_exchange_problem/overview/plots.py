"""Retain trace, sampled histogram and best-observation bipartite graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .model import WEIGHTS
from .workflow import Workflow


def plot_results(workflow: Workflow, directory: Path) -> tuple[Path, Path, Path]:
    """Render actual observations without replacing an infeasible result."""
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / "cost-convergence.png"
    histogram = directory / "sampled-costs.png"
    matching = directory / "best-observed-matching.png"
    figure, axes = plt.subplots()
    axes.plot([row.cost for row in workflow.trace])
    axes.set_xlabel("Objective evaluation")
    axes.set_ylabel("CVaR 0.7 penalized cost")
    axes.set_title("Kidney matching cost convergence")
    figure.tight_layout()
    figure.savefig(trace)
    plt.close(figure)
    figure, axes = plt.subplots()
    shots = sum(row.count for row in workflow.samples)
    axes.hist(
        [row.assignment.penalized_cost for row in workflow.samples],
        weights=[row.count / shots for row in workflow.samples],
        bins=30,
        edgecolor="black",
    )
    axes.set_xlabel("Penalized cost")
    axes.set_ylabel("Observed probability")
    axes.set_title("Histogram of cost weighted by observed probability")
    figure.tight_layout()
    figure.savefig(histogram)
    plt.close(figure)
    best = workflow.best_observed.assignment
    donors = tuple(f"donor{i + 1}" for i in range(3))
    patients = tuple(f"patient{i + 1}" for i in range(3))
    graph = nx.DiGraph()
    graph.add_nodes_from((*donors, *patients))
    for donor in range(3):
        for patient in range(3):
            if best.rows[donor][patient]:
                graph.add_edge(
                    donors[donor],
                    patients[patient],
                    weight=WEIGHTS[3 * donor + patient],
                )
    positions = {name: (0, -index) for index, name in enumerate(donors)} | {
        name: (1, -index) for index, name in enumerate(patients)
    }
    figure, axes = plt.subplots(figsize=(10, 6))
    nx.draw_networkx(
        graph,
        positions,
        ax=axes,
        node_color=["#F43764"] * 3 + ["#119DA4"] * 3,
        node_size=1200,
        arrows=True,
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        ax=axes,
        edge_labels=nx.get_edge_attributes(graph, "weight"),
        label_pos=0.6,
    )
    axes.set_title(
        f"Best observed assignment: score {best.score:.2f}, feasible={best.feasible}"
    )
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(matching)
    plt.close(figure)
    return trace, histogram, matching
