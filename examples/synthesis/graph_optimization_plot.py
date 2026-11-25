"""Source workflow figures for the two fixed native subset-selection examples."""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure

from .graph_optimization import SubsetWorkflow


def plot_workflow(
    workflow: SubsetWorkflow,
    edges: tuple[tuple[int, int], ...],
    *,
    seed: int,
    title: str,
) -> Figure:
    """Show the graph, actual objective calls, samples, and both feasible answers.

    A deterministic force layout replaces the source Kamada-Kawai layout without
    adding SciPy. Histograms use unpenalized negative size, including infeasible
    observations; selected answers are filtered for feasibility separately.
    """
    width = len(workflow.program.registers["v"])
    graph = nx.Graph()
    graph.add_nodes_from(range(width))
    graph.add_edges_from(edges)
    positions = nx.spring_layout(graph, seed=seed, method="force")
    figure = plt.figure(figsize=(15, 8), layout="constrained")
    axes = figure.subplots(2, 3)
    best = workflow.best_feasible
    answers = (
        (axes[0, 0], (), "Original seeded graph"),
        (axes[1, 0], best.nodes if best else (), "Best observed feasible subset"),
        (axes[1, 1], workflow.exact_nodes, "Exact bounded classical reference"),
    )
    for axis, nodes, label in answers:
        nx.draw_networkx(
            graph,
            pos=positions,
            ax=axis,
            node_size=550,
            node_color=["#E39474" if i in nodes else "#AFC4DC" for i in range(width)],
        )
        axis.set_title(f"{label}: {list(nodes)}" if nodes else label)
        axis.set_axis_off()
    axes[0, 1].plot([entry.cost for entry in workflow.trace], color="#31568A")
    axes[0, 1].set(
        xlabel="Actual quantum objective evaluation",
        ylabel="Lower-tail penalized cost",
        title="CVaR 0.7 optimization trace",
    )
    for rows, label, color in (
        (workflow.optimized, "optimized", "#31568A"),
        (workflow.uniform, "uniform", "#D69545"),
    ):
        shots = sum(row.count for row in rows)
        axes[0, 2].hist(
            [-len(row.nodes) for row in rows],
            bins=[i - 0.5 for i in range(-width, 2)],
            weights=[row.count / shots for row in rows],
            alpha=0.6,
            label=label,
            color=color,
        )
    axes[0, 2].set(
        xlabel="Unpenalized cost: negative subset size",
        ylabel="Empirical probability",
        title="Local sampled distributions",
    )
    axes[0, 2].legend()
    axes[1, 2].set_axis_off()
    axes[1, 2].text(
        0.05,
        0.8,
        f"Iterations: {workflow.optimizer.iterations}\nActual evaluations: {workflow.optimizer.evaluations}\nConverged: {workflow.optimizer.converged}\nFeasible sampled answer: {list(best.nodes) if best else None}\nExact reference: {list(workflow.exact_nodes)}\nLocal CPU samples; no hardware claim",
        va="top",
        fontsize=11,
    )
    figure.suptitle(title, fontsize=15)
    return figure
