"""Preserve source graph, trace, histogram and actual/reference cover figure roles."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from .workflow import Workflow, model_table, source_edges


def cover_figure(workflow: Workflow) -> Figure:
    """Draw the original graph and actual readout with a separate exact reference."""
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(source_edges())
    positions = nx.spring_layout(graph, seed=2718)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    nx.draw(graph, positions, with_labels=True, node_color="lightblue", ax=axes[0, 0])
    axes[0, 0].set_title("Original ten-node graph; K=5")
    axes[0, 1].plot([entry.cost for entry in workflow.trace])
    axes[0, 1].set(
        xlabel="Actual quantum evaluation",
        ylabel="Lower-tail CVaR 0.7",
        title=f"Bounded native optimization; converged={workflow.optimizer.converged}",
    )
    maximum = max(row.penalized_cost for row in (*workflow.samples, *workflow.uniform))
    bins = np.linspace(0, maximum + 1, 41)
    for label, samples in (
        ("Optimized", workflow.samples),
        ("Uniform", workflow.uniform),
    ):
        total = sum(row.count for row in samples)
        axes[0, 2].hist(
            [row.penalized_cost for row in samples],
            bins=bins,
            weights=[row.count / total for row in samples],
            alpha=0.6,
            label=label,
        )
    axes[0, 2].legend()
    axes[0, 2].set(
        xlabel="Penalized uncovered-edge cost",
        ylabel="Empirical probability",
        title="Actual seeded observations; common histogram bins",
    )
    best = workflow.best_feasible
    for axis, word, title in (
        (
            axes[1, 0],
            best.word if best is not None else None,
            "Best observed feasible cover",
        ),
        (
            axes[1, 1],
            workflow.exact_reference_word,
            "Exact bounded classical reference",
        ),
    ):
        if word is None:
            axis.axis("off")
            axis.text(0.1, 0.5, "No feasible assignment was observed")
            continue
        selected = {node for node in range(10) if word & (1 << node)}
        nx.draw(
            graph,
            positions,
            with_labels=True,
            node_color=[
                "tomato" if node in selected else "lightblue" for node in graph
            ],
            edge_color=[
                "gold" if a in selected or b in selected else "gray"
                for a, b in graph.edges()
            ],
            ax=axis,
        )
        axis.set_title(
            f"{title}\nNodes={sorted(selected)}; uncovered={model_table()[word][0]}"
        )
    axes[1, 2].axis("off")
    feasible = sum(row.count for row in workflow.samples if row.feasible)
    uniform = sum(row.count for row in workflow.uniform if row.feasible)
    axes[1, 2].text(
        0.05,
        0.9,
        f"Actual feasible observations\nOptimized: {feasible}/{sum(row.count for row in workflow.samples)}"
        f"\nUniform: {uniform}/{sum(row.count for row in workflow.uniform)}\n\n"
        "Independent reference after sampling\nSeeded spring layout replaces source Kamada-Kawai.\n"
        "No convergence or quantum advantage claim.",
        va="top",
        fontsize=10,
    )
    return figure
