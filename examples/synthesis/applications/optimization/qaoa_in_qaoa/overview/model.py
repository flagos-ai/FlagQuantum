"""Source-sized fresh graph, bounded communities and exact signed contraction."""

import math
import random
from itertools import combinations, product

import networkx as nx

MAX_NODES = 6


def fresh_graph() -> tuple[nx.Graph, random.Random]:
    generator = random.Random(246)
    graph = nx.erdos_renyi_graph(40, 0.2, seed=generator)
    return graph, generator


def random_split(graph: nx.Graph, generator: random.Random) -> list[set[int]]:
    remaining = list(graph.nodes)
    result = []
    while remaining:
        group = set(generator.sample(sorted(remaining), min(len(remaining), MAX_NODES)))
        result.append(group)
        remaining = list(set(remaining) - group)
    return result


def split_communities(
    graph: nx.Graph, generator: random.Random, level: int = 0
) -> list[set[int]]:
    try:
        initial = list(nx.community.greedy_modularity_communities(graph))
    except (ValueError, ZeroDivisionError, nx.NetworkXError):
        return random_split(graph, generator)
    result: list[set[int]] = []
    for values in initial:
        group = set(values)
        if level > 100 and len(group) > MAX_NODES:
            while group:
                if len(group) > MAX_NODES:
                    chosen = set(
                        generator.sample(
                            sorted(group), min(math.ceil(len(group) / 2), MAX_NODES)
                        )
                    )
                else:
                    chosen = set(group)
                result.append(chosen)
                group -= chosen
        elif len(group) > MAX_NODES:
            result.extend(
                split_communities(graph.subgraph(group), generator, level + 1)
            )
        else:
            result.append(group)
    return result


def cut_value(graph: nx.Graph, solution: dict[int, int]) -> float:
    return sum(
        float(data.get("weight", 1.0))
        for first, second, data in graph.edges(data=True)
        if solution[first] != solution[second]
    )


def contract(
    graph: nx.Graph, groups: list[set[int]], solution: dict[int, int]
) -> nx.Graph:
    result = nx.Graph()
    result.add_nodes_from(range(len(groups)))
    for first, second in combinations(range(len(groups)), 2):
        weight = sum(
            float(graph[a][b].get("weight", 1.0))
            * (1 if solution[a] == solution[b] else -1)
            for a, b in product(groups[first], groups[second])
            if graph.has_edge(a, b)
        )
        if weight:
            result.add_edge(first, second, weight=weight)
    return result


def expand_groups(
    previous: dict[int, list[int]], groups: list[set[int]]
) -> dict[int, list[int]]:
    return {
        index: [
            original
            for node in group
            for original in (previous[node] if previous else [node])
        ]
        for index, group in enumerate(groups)
    }


def local_edges(graph: nx.Graph) -> tuple[tuple[int, int, float], ...]:
    nodes = tuple(graph.nodes)
    indices = {node: index for index, node in enumerate(nodes)}
    return tuple(
        (
            min(indices[a], indices[b]),
            max(indices[a], indices[b]),
            float(data.get("weight", 1.0)),
        )
        for a, b, data in graph.edges(data=True)
    )
