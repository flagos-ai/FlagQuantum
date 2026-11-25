"""Regenerate the source six-qubit glued trees and full Pauli expansion."""

import itertools
import random

import networkx as nx
import torch

from examples.synthesis.pauli_evolution import Term

PAULIS = {
    "I": torch.eye(2, dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.diag(torch.tensor([1, -1], dtype=torch.complex128)),
}


def fresh_matrix() -> tuple[torch.Tensor, int]:
    dim = 4
    left = nx.balanced_tree(2, dim - 1)
    right = nx.relabel_nodes(left, lambda node: 2 ** (dim + 1) - 3 - node)
    graph = nx.union(left, right)
    counts = dict.fromkeys(range(2**dim - 1, 2 ** (dim - 1) + 2**dim - 1), 0)
    rng = random.Random(0)
    for node in range(2 ** (dim - 1) - 1, 2**dim - 1):
        candidates = [i for i, count in counts.items() if count < 1]
        if not candidates:
            candidates = [i for i, count in counts.items() if count < 2]
        for neighbor in rng.sample(candidates, 2):
            counts[neighbor] += 1
            graph.add_edge(node, neighbor)
    adjacency = torch.tensor(
        nx.to_numpy_array(graph, nodelist=sorted(graph.nodes)), dtype=torch.float64
    )
    left_block = 3 * torch.eye(30, dtype=torch.float64) - adjacency
    block = torch.cat(
        (torch.linalg.cholesky(left_block), torch.zeros((30, 4), dtype=torch.float64)),
        dim=1,
    )
    matrix = torch.zeros((64, 64), dtype=torch.complex128)
    matrix[:30, 30:] = -block
    matrix[30:, :30] = -block.T
    return matrix, graph.number_of_edges()


def decompose(matrix: torch.Tensor) -> tuple[tuple[Term, ...], float]:
    complete = []
    reconstruction = torch.zeros_like(matrix)
    for labels in itertools.product("IXYZ", repeat=6):
        factor = PAULIS[labels[0]]
        for label in labels[1:]:
            factor = torch.kron(factor, PAULIS[label])
        coef = (factor.conj() * matrix).sum() / 64
        assert abs(float(coef.imag)) < 1e-14
        weight = float(coef.real)
        if abs(weight) > 1e-14:
            reconstruction += weight * factor
            support = tuple(
                (5 - i, axis) for i, axis in enumerate(labels) if axis != "I"
            )
            complete.append((weight, support))
    ordered = tuple(sorted(complete, key=lambda t: abs(t[0]), reverse=True))
    err = float((reconstruction - matrix).abs().max())
    assert err < 1e-12
    return ordered, err
