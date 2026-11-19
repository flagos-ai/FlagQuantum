"""Fresh domination model and exact cost table for the original six-node graph."""

from dataclasses import dataclass

import networkx as nx
import torch

REGISTER_NAMES = ("x", "s0", "s1", "s2", "s3", "s4", "s5")
PHYSICAL_WIDTH = 20
DECISIONS = 6
PENALTY = 10
QUANTILE = 0.7


@dataclass(frozen=True)
class Model:
    graph: nx.Graph
    neighborhoods: tuple[tuple[int, ...], ...]
    slack_weights: tuple[tuple[int, ...], ...]

    def dominates(self, word: int) -> bool:
        return all(
            any(word & (1 << vertex) for vertex in group)
            for group in self.neighborhoods
        )


def fresh_model() -> Model:
    graph = nx.erdos_renyi_graph(n=DECISIONS, p=0.6, seed=8)
    groups = tuple(
        tuple(sorted({vertex, *graph.neighbors(vertex)})) for vertex in graph
    )
    weights = tuple((1, 2) if len(group) == 4 else (1, 2, 1) for group in groups)
    return Model(graph, groups, weights)


def energies(model: Model) -> torch.Tensor:
    """Assign full penalized energy to every joint decision/slack basis word."""
    words = torch.arange(1 << PHYSICAL_WIDTH, dtype=torch.int64)
    decision_bits = tuple((words >> bit) & 1 for bit in range(DECISIONS))
    total = words.new_zeros(words.shape)
    for bit_values in decision_bits:
        total += bit_values
    offset = DECISIONS
    for group, weights in zip(model.neighborhoods, model.slack_weights, strict=True):
        residual = words.new_ones(words.shape)
        for j, weight in enumerate(weights):
            residual += weight * ((words >> (offset + j)) & 1)
        for vertex in group:
            residual -= decision_bits[vertex]
        total += PENALTY * residual.square()
        offset += len(weights)
    if offset != PHYSICAL_WIDTH:
        raise ValueError("model must match the original twenty-bit encoding")
    return total


def exact_solutions(model: Model) -> tuple[int, ...]:
    feasible = [word for word in range(1 << DECISIONS) if model.dominates(word)]
    optimum = min(word.bit_count() for word in feasible)
    return tuple(word for word in feasible if word.bit_count() == optimum)
