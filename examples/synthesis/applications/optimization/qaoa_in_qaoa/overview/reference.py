"""Independent complete cut tables and dense bounded Circuit phase reference."""

import torch

import flagquantum as fq


def reference_circuit(
    node_count: int,
    edges: tuple[tuple[int, int, float], ...],
    parameters: tuple[float, ...],
) -> fq.Circuit:
    costs = torch.tensor(
        [
            sum(
                weight
                for first, second, weight in edges
                if ((word >> first) & 1) != ((word >> second) & 1)
            )
            for word in range(1 << node_count)
        ],
        dtype=torch.float64,
    )
    circuit = fq.Circuit(node_count, dtype=torch.complex128)
    for wire in range(node_count):
        circuit.h(wire)
    for layer in range(2):
        circuit.any(
            *range(node_count),
            matrix=torch.diag(torch.exp(1j * parameters[2 * layer] * costs))
        )
        for wire in range(node_count):
            circuit.rx(wire, theta=parameters[2 * layer + 1])
    return circuit
