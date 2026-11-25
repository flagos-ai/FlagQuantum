"""Show source graph, actual optimization, observations and observed minimum coloring."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .workflow import EDGES, Sample, Workflow


def _draw_graph(axis: Axes, row: Sample | None = None) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(EDGES)
    positions = nx.circular_layout(graph)
    palette = ("cornflowerblue", "coral", "mediumseagreen")
    colors = [
        (
            "lightgray"
            if row is None or len(row.node_colors[v]) != 1
            else palette[row.node_colors[v][0]]
        )
        for v in range(5)
    ]
    nx.draw_networkx(graph, pos=positions, node_color=colors, ax=axis)
    axis.set_axis_off()


def optimization_figure(workflow: Workflow) -> Figure:
    """Retain four figure roles with common fifty-bin empirical cost histograms."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    _draw_graph(axes[0, 0])
    axes[0, 0].set_title("Source graph: five nodes, five edges, three colors")
    axes[0, 1].plot([row.cost for row in workflow.trace])
    axes[0, 1].set(
        xlabel="Actual quantum evaluation",
        ylabel="Retained-cost lower-tail CVaR 0.7",
        title=f"Native bounded search; converged={workflow.optimizer.converged}",
    )
    rows = (*workflow.samples, *workflow.uniform)
    bins = np.linspace(
        min(row.retained_quantum_cost for row in rows),
        max(row.retained_quantum_cost for row in rows) + 1,
        51,
    )
    for label, samples in (
        ("Optimized", workflow.samples),
        ("Uniform", workflow.uniform),
    ):
        total = sum(row.count for row in samples)
        axes[1, 0].hist(
            [row.retained_quantum_cost for row in samples],
            bins=bins,
            weights=[row.count / total for row in samples],
            alpha=0.6,
            label=label,
        )
    axes[1, 0].legend()
    axes[1, 0].set(
        xlabel="Retained quantum penalty cost",
        ylabel="Empirical probability",
        title="Actual observations; common fifty-bin histograms",
    )
    best = workflow.best_feasible
    if best is None:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.1, 0.5, "No feasible coloring was observed")
    else:
        _draw_graph(axes[1, 1], best)
        axes[1, 1].set_title(
            f"Best observed feasible coloring: {best.objective} colors"
        )
    optimized = sum(row.count for row in workflow.samples if row.feasible)
    uniform = sum(row.count for row in workflow.uniform if row.feasible)
    figure.suptitle(f"Feasible observations: {optimized} optimized, {uniform} uniform")
    return figure
