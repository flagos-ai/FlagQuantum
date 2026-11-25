"""Render original supply network, actual cost distributions and assignment graphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .model import CUSTOMERS, FACILITIES, WEIGHTS, Assignment
from .workflow import Workflow


def plot_network(
    path: Path,
    value: Assignment | None = None,
    *,
    label: str = "Facility/customer network",
) -> Path:
    """Draw every facility/customer and only the chosen edges for an assignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = FACILITIES
    consumers = CUSTOMERS
    graph = nx.DiGraph()
    graph.add_nodes_from((*sources, *consumers))
    for i, source in enumerate(sources):
        for j, consumer in enumerate(consumers):
            if value is None or value.rows[i][j]:
                graph.add_edge(source, consumer, weight=WEIGHTS[4 * i + j])
    positions = {name: (0.0, -i / 2) for i, name in enumerate(sources)} | {
        name: (1.0, -i / 3) for i, name in enumerate(consumers)
    }
    figure, axes = plt.subplots(figsize=(10, 6))
    colors = [
        ("#F43764" if value is None or any(value.rows[i]) else "#F4F9E9")
        for i in range(3)
    ] + ["#119DA4"] * 4
    nx.draw_networkx(
        graph, positions, ax=axes, node_color=colors, node_size=1100, arrows=True
    )
    labels = nx.get_edge_attributes(graph, "weight")
    for source, position in zip(sources, (0.8, 0.5, 0.2), strict=True):
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={
                edge: cost for edge, cost in labels.items() if edge[0] == source
            },
            ax=axes,
            label_pos=position,
            font_size=10,
        )
    suffix = (
        ""
        if value is None
        else f": raw cost {value.raw_cost:.2f}, feasible={value.feasible}"
    )
    axes.set_title(label + suffix)
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_results(
    workflow: Workflow, directory: Path
) -> tuple[Path, Path, Path, Path, Path]:
    """Keep finite optimized/uniform costs, including infeasible observations."""
    directory.mkdir(parents=True, exist_ok=True)
    original = plot_network(directory / "facility-network.png")
    trace = directory / "cost-convergence.png"
    figure, axes = plt.subplots()
    axes.plot([row.cost for row in workflow.trace])
    axes.set_xlabel("Objective evaluation")
    axes.set_ylabel("Exact mean penalized cost")
    axes.set_title("Facility-location cost convergence")
    figure.tight_layout()
    figure.savefig(trace)
    plt.close(figure)
    histogram = directory / "sampled-costs.png"
    figure, axes = plt.subplots()
    for label, samples in (
        ("optimized", workflow.samples),
        ("uniform", workflow.uniform),
    ):
        shots = sum(row.count for row in samples)
        axes.hist(
            [row.assignment.cost for row in samples],
            weights=[row.count / shots for row in samples],
            bins=50,
            edgecolor="black",
            alpha=0.6,
            label=label,
        )
    axes.legend()
    axes.set_xlabel("Joint penalized cost")
    axes.set_ylabel("Observed probability")
    axes.set_title("Actual optimized and zero-angle uniform samples")
    figure.tight_layout()
    figure.savefig(histogram)
    plt.close(figure)
    observed = plot_network(
        directory / "best-observed-facilities.png",
        workflow.best_observed.assignment,
        label="Best observed facilities",
    )
    reference = plot_network(
        directory / "classical-reference-facilities.png",
        workflow.reference[0],
        label="Separate exhaustive classical solution",
    )
    return original, trace, histogram, observed, reference
