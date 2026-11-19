"""Native patch selection from the complete original bipartite attack graph."""

import math
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import networkx as nx

from examples.synthesis.graph_optimization import (
    SubsetWorkflow,
    run_subset_optimization,
)
from flagquantum import synth

from ..main import main

if TYPE_CHECKING:
    from matplotlib.figure import Figure

INCIDENCE = (
    ("A", "B", "D", "F"),
    ("A", "B"),
    ("A", "D", "E"),
    ("B", "C", "F"),
    ("G",),
    ("F", "G"),
    ("B", "C", "F"),
    ("C", "D", "G"),
)


def source_graph() -> nx.Graph:
    """Retain all eight resources, seven vulnerabilities and original incidences."""
    graph = nx.Graph()
    graph.add_nodes_from(range(8), bipartite=0)
    graph.add_nodes_from("ABCDEFG", bipartite=1)
    graph.add_edges_from((i, v) for i, values in enumerate(INCIDENCE) for v in values)
    return graph


def kill_chain_graph() -> nx.Graph:
    """Retain the source's six-node illustrative attack chain."""
    graph = nx.Graph()
    graph.add_nodes_from((0, 3, 7), bipartite=0)
    graph.add_nodes_from(("B", "C", "F"), bipartite=1)
    graph.add_edges_from(((0, "B"), (0, "F"), (3, "F"), (3, "C"), (7, "C")))
    return graph


def dual_graph() -> nx.Graph:
    """Join resources that share a vulnerability, explicitly selecting resources."""
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(
        (i, j)
        for i, j in combinations(range(8), 2)
        if set(INCIDENCE[i]).intersection(INCIDENCE[j])
    )
    return graph


def model_tables() -> tuple[tuple[float, ...], frozenset[int]]:
    """Evaluate every original cover assignment, retaining penalty 10."""
    edges = tuple(dual_graph().edges())
    costs = []
    feasible = set()
    for word in range(256):
        uncovered = sum(not word & (1 << i) and not word & (1 << j) for i, j in edges)
        costs.append(float(word.bit_count() + 10 * uncovered))
        if uncovered == 0:
            feasible.add(word)
    return tuple(costs), frozenset(feasible)


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    """Bind the unchanged original eight-qubit three-layer native quantum model."""
    if len(parameters) != 6 or any(
        isinstance(value, bool) or not math.isfinite(value) for value in parameters
    ):
        raise ValueError("three patching layers require six finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=8,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=4 * 1024**2,
    )


def run_workflow(
    *, maxiter: int = 60, shots: int = 1000, seed: int = 1916
) -> SubsetWorkflow:
    """Search actual CVaR0.9 quantum distributions and rank minimum-size covers.

    The hosted source leaves optimizer details and sample counts implicit. Reuse
    the explicit native Nelder-Mead workflow and seeded local observations; no
    claim equates its trajectory to the hosted implementation.
    """
    costs, feasible = model_tables()
    workflow = run_subset_optimization(
        build_program,
        costs,
        feasible,
        maxiter=maxiter,
        shots=shots,
        seed=seed,
        quantile=0.9,
    )
    # The shared subset tutorial selects maximum size. A cover minimizes size.
    return replace(
        workflow,
        optimized=tuple(
            sorted(
                workflow.optimized,
                key=lambda row: (not row.feasible, len(row.nodes), row.word),
            )
        ),
        uniform=tuple(
            sorted(
                workflow.uniform,
                key=lambda row: (not row.feasible, len(row.nodes), row.word),
            )
        ),
    )


def remaining_graph(nodes: tuple[int, ...]) -> nx.Graph:
    """Remove the actually observed patch set from the original attack graph."""
    if len(set(nodes)) != len(nodes) or any(
        type(i) is not int or not 0 <= i < 8 for i in nodes
    ):
        raise ValueError("patch set must contain distinct resource nodes in 0..7")
    graph = source_graph()
    graph.remove_nodes_from(nodes)
    return graph


def plot_graph(
    graph: nx.Graph,
    title: str,
    *,
    layout: Literal["spring", "bipartite", "circular"] = "spring",
    selected: tuple[int, ...] = (),
) -> "Figure":
    """Render source graph views with deterministic layouts and actual selections."""
    from matplotlib import pyplot as plt

    if layout == "bipartite":
        left = tuple(
            i for i, data in graph.nodes(data=True) if data.get("bipartite") == 0
        )
        positions = nx.bipartite_layout(graph, left)
    elif layout == "circular":
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=1916, method="force")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=axis,
        node_color=["#E39474" if i in selected else "#AFC4DC" for i in graph.nodes],
        edge_color=[
            "#D69545" if i in selected or j in selected else "#888888"
            for i, j in graph.edges
        ],
        node_size=600,
    )
    axis.set_title(title)
    axis.set_axis_off()
    return figure


def plot_trace(workflow: SubsetWorkflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot([row.cost for row in workflow.trace])
    axis.set(
        xlabel="Objective evaluations",
        ylabel="Penalized cost CVaR (0.9)",
        title="Native patch selection cost search",
    )
    return figure


def plot_costs(workflow: SubsetWorkflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, rows in (
        ("optimized", workflow.optimized),
        ("uniform", workflow.uniform),
    ):
        shots = sum(row.count for row in rows)
        axis.hist(
            [len(row.nodes) for row in rows],
            bins=[i - 0.5 for i in range(10)],
            weights=[row.count / shots for row in rows],
            alpha=0.6,
            label=label,
        )
    axis.set(
        xlabel="Selected resource count",
        ylabel="Observed probability",
        title="Sampled patch counts (including infeasible covers)",
    )
    axis.legend()
    return figure
