"""Render the actual original forty-node solution with seeded local layout."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .workflow import Workflow


def plot_solution(workflow: Workflow, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = nx.spring_layout(workflow.graph, seed=np.random.RandomState(4812))
    nodes = ["royalblue" if bit == 0 else "tomato" for bit in workflow.assignment]
    edges = [
        "black" if workflow.assignment[a] == workflow.assignment[b] else "darkorange"
        for a, b in workflow.graph.edges
    ]
    figure, axes = plt.subplots(figsize=(12, 8))
    nx.draw(
        workflow.graph,
        positions,
        node_color=nodes,
        edge_color=edges,
        with_labels=True,
        width=1.3,
        node_size=280,
        font_size=8,
        ax=axes,
    )
    axes.set_title(
        f"Observed hierarchical QAOA cut {workflow.cut:g}; greedy {workflow.greedy_cut:g}; random {workflow.random_cut:g}"
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
