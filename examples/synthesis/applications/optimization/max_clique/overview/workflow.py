"""Complete local max clique optimization workflow."""

import math

import networkx as nx

from examples.synthesis.graph_optimization import (
    SubsetWorkflow,
    run_subset_optimization,
)
from flagquantum import synth

from ..main import main


def source_edges() -> tuple[tuple[int, int], ...]:
    """Construct the original seeded graph with its original node order."""
    graph = nx.erdos_renyi_graph(7, 0.6, seed=79)
    return tuple(sorted((int(a), int(b)) for a, b in graph.edges()))


def model_tables() -> tuple[tuple[float, ...], frozenset[int]]:
    """Evaluate the retained cost and graph feasibility on every bounded assignment."""
    graph = nx.Graph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(source_edges())
    forbidden = tuple(nx.complement(graph).edges())
    costs = []
    feasible = set()
    for word in range(1 << 7):
        nodes = {i for i in range(7) if word & (1 << i)}
        conflicts = sum(a in nodes and b in nodes for a, b in forbidden)
        costs.append(float(-len(nodes) + 4 * conflicts**2))
        if conflicts == 0:
            feasible.add(word)
    return tuple(costs), frozenset(feasible)


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    """Reuse the retained native qfunc with explicit finite parameters and budgets."""
    if len(parameters) != 6 or any(not math.isfinite(value) for value in parameters):
        raise ValueError("three layers require six finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=7,
        max_workspace=0,
        max_operations=5000,
        memory_limit_bytes=4 * 1024**2,
    )


def run_workflow(
    *, maxiter: int = 50, shots: int = 1000, seed: int = 2718
) -> SubsetWorkflow:
    """Optimize the actual quantum program and report feasible local observations."""
    costs, feasible = model_tables()
    return run_subset_optimization(
        build_program, costs, feasible, maxiter=maxiter, shots=shots, seed=seed
    )
