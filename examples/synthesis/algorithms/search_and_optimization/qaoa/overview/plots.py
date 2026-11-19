"""Preserve the QAOA tutorial's graph, trace and sampled-solution figure roles."""

import math

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure

from .workflow import EDGES, Workflow


def maxcut_figure(workflow: Workflow) -> Figure:
    """Show the original graph, measured trace, and every best sampled cut."""
    if workflow.case != "maxcut":
        raise ValueError("MaxCut plots require the MaxCut workflow")
    if not workflow.samples:
        raise ValueError("MaxCut plots require actual observations")
    best_cost = min(row.cost for row in workflow.samples)
    best = tuple(
        sorted(
            (row for row in workflow.samples if abs(row.cost - best_cost) < 1e-3),
            key=lambda row: (-row.count, row.assignment),
        )
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(EDGES)
    positions = nx.spring_layout(graph, seed=2718)
    rows = 1 + math.ceil(len(best) / 2)
    figure = plt.figure(figsize=(12, 4 * rows), constrained_layout=True)
    graph_axis = figure.add_subplot(rows, 2, 1)
    nx.draw(graph, positions, with_labels=True, ax=graph_axis, node_color="lightblue")
    graph_axis.set_title("Original five-node graph")
    trace_axis = figure.add_subplot(rows, 2, 2)
    trace_axis.plot([row.cost for row in workflow.trace])
    trace_axis.set_xlabel("Actual quantum evaluation")
    trace_axis.set_ylabel("Expected negative cut fraction")
    trace_axis.set_title(
        f"Bounded native optimization; converged={workflow.optimizer.converged}"
    )
    total = sum(row.count for row in workflow.samples)
    for index, sample in enumerate(best):
        axis = figure.add_subplot(rows, 2, index + 3)
        (word,) = sample.assignment
        colors = [
            "tomato" if word & (1 << node) else "cornflowerblue" for node in range(5)
        ]
        nx.draw(graph, positions, with_labels=True, ax=axis, node_color=colors)
        axis.set_title(
            f"Observed cut {index + 1}: {round(-sample.cost * len(EDGES))} edges\n"
            f"Count={sample.count}/{total}; cost={sample.cost:.3f}"
        )
    return figure


def knapsack_figure(workflow: Workflow) -> Figure:
    """Show the measured cost trajectory and its observed named assignments."""
    if workflow.case != "knapsack":
        raise ValueError("Knapsack plots require the Knapsack workflow")
    figure = plt.figure(figsize=(12, 4), constrained_layout=True)
    axis = figure.add_subplot(1, 2, 1)
    axis.plot([row.cost for row in workflow.trace])
    axis.set_xlabel("Actual quantum evaluation")
    axis.set_ylabel("Expected constrained negative value")
    axis.set_title(
        f"Bounded native optimization; converged={workflow.optimizer.converged}"
    )
    table_axis = figure.add_subplot(1, 2, 2)
    table_axis.axis("off")
    table = table_axis.table(
        cellText=[
            [
                str(row.assignment),
                str(round(-row.cost)),
                str(row.feasible),
                str(row.count),
            ]
            for row in workflow.samples[:10]
        ],
        colLabels=["(a, b)", "Objective value", "Feasible", "Observed count"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    table_axis.set_title("Actual seeded samples; infeasible objective is zero")
    return figure
