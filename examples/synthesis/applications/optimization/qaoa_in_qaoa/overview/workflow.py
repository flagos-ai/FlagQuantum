"""Run fresh bounded quantum subproblems and signed hierarchical graph flips."""

import random
from dataclasses import dataclass

import networkx as nx
import torch

from examples.synthesis.graph_optimization import Evaluation, lower_tail_cost
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.compiler import ResourceEstimate

from .model import (
    MAX_NODES,
    contract,
    cut_value,
    expand_groups,
    fresh_graph,
    local_edges,
    split_communities,
)
from .quantum import build_program, read_distribution


def initial_parameters(edges: tuple[tuple[int, int, float], ...]) -> tuple[float, ...]:
    magnitudes = sorted(abs(weight) / 2 for _, _, weight in edges)
    largest = magnitudes[len(magnitudes) // 2 :]
    time_step = len(largest) / (2 * sum(largest))
    return 0.0, time_step, time_step, 0.0


class _ObjectiveBudgetError(Exception):
    """Stop before a community exceeds twenty actual quantum objective queries."""


@dataclass(frozen=True)
class Observation:
    word: int
    cut: float
    count: int


@dataclass(frozen=True)
class CommunityRun:
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int, float], ...]
    assignment: tuple[int, ...]
    trace: tuple[Evaluation, ...]
    observations: tuple[Observation, ...]
    parameters: tuple[float, ...]
    budget_reached: bool
    converged: bool
    resources: ResourceEstimate | None


def solve_community(graph: nx.Graph, generator: random.Random) -> CommunityRun:
    nodes = tuple(graph.nodes)
    if not nodes or len(nodes) > MAX_NODES:
        raise ValueError("quantum communities must contain one to six nodes")
    edges = local_edges(graph)
    if not edges:
        bits = tuple(generator.choice((0, 1)) for _ in nodes)
        return CommunityRun(nodes, edges, bits, (), (), (), False, False, None)
    cuts = tuple(
        sum(
            weight
            for first, second, weight in edges
            if ((word >> first) & 1) != ((word >> second) & 1)
        )
        for word in range(1 << len(nodes))
    )
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        if len(trace) >= 20:
            raise _ObjectiveBudgetError()
        values = tuple(float(value) for value in parameters)
        cost = lower_tail_cost(
            read_distribution(build_program(len(nodes), edges, values)),
            tuple(-value for value in cuts),
            1.0,
        )
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    budget_reached = False
    converged = False
    try:
        optimized = CobylaOptimizer(maxiter=100, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(initial_parameters(edges), dtype=torch.float64)
        )
        converged = optimized.converged
    except _ObjectiveBudgetError:
        budget_reached = True
    parameters = min(trace, key=lambda value: value.cost).parameters
    program = build_program(len(nodes), edges, parameters)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    draws = torch.multinomial(
        torch.tensor(read_distribution(program), dtype=torch.float64),
        1000,
        replacement=True,
        generator=torch.Generator().manual_seed(10),
    )
    counts = torch.bincount(draws, minlength=len(cuts))
    observations = tuple(
        sorted(
            (
                Observation(word, cuts[word], int(count))
                for word, count in enumerate(counts)
                if int(count)
            ),
            key=lambda value: (value.cut, value.count),
            reverse=True,
        )
    )
    selected = observations[0]
    bits = tuple((selected.word >> index) & 1 for index in range(len(nodes)))
    return CommunityRun(
        nodes,
        edges,
        bits,
        tuple(trace),
        observations,
        parameters,
        budget_reached,
        converged,
        resources,
    )


@dataclass(frozen=True)
class Workflow:
    graph: nx.Graph
    assignment: tuple[int, ...]
    cut: float
    greedy_cut: float
    random_cut: float
    runs: tuple[CommunityRun, ...]
    initial_groups: tuple[tuple[int, ...], ...]
    contraction_sizes: tuple[int, ...]


def run_workflow() -> Workflow:
    graph, generator = fresh_graph()
    groups = split_communities(graph, generator)
    initial_groups = tuple(tuple(sorted(group)) for group in groups)
    full: dict[int, int] = {}
    runs: list[CommunityRun] = []

    def solve_groups(current: nx.Graph, communities: list[set[int]]) -> dict[int, int]:
        solution: dict[int, int] = {}
        for group in communities:
            result = solve_community(current.subgraph(group), generator)
            runs.append(result)
            solution.update(zip(result.nodes, result.assignment, strict=True))
        return solution

    full.update(solve_groups(graph, groups))
    compressed = contract(graph, groups, full)
    basic = expand_groups({}, groups)
    sizes = [len(compressed)]
    while len(compressed) > MAX_NODES:
        groups = split_communities(compressed, generator)
        if len(groups) >= len(compressed):
            raise ValueError("community contraction did not reduce the graph")
        intermediate = solve_groups(compressed, groups)
        for node, bit in intermediate.items():
            if bit:
                for original in basic[node]:
                    full[original] ^= 1
        compressed = contract(compressed, groups, intermediate)
        basic = expand_groups(basic, groups)
        sizes.append(len(compressed))
    final = solve_community(compressed, generator)
    runs.append(final)
    for node, bit in zip(final.nodes, final.assignment, strict=True):
        if bit:
            for original in basic[node]:
                full[original] ^= 1
    greedy = nx.algorithms.approximation.maxcut.one_exchange(graph, seed=generator)[0]
    random_cut = nx.algorithms.approximation.maxcut.randomized_partitioning(
        graph, seed=generator
    )[0]
    return Workflow(
        graph,
        tuple(full[node] for node in graph.nodes),
        cut_value(graph, full),
        float(greedy),
        float(random_cut),
        tuple(runs),
        initial_groups,
        tuple(sizes),
    )
