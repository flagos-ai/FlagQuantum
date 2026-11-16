"""Independent dense-oracle and reflection comparison for the graph search."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.grover.main import (
    EDGES,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def marked(bits: tuple[int, ...]) -> bool:
    cut_size = sum(bits[left] != bits[right] for left, right in EDGES)
    return cut_size >= 4


def reference_circuit(iterations: int) -> fq.Circuit:
    width = 5
    size = 1 << width
    diagonal = torch.tensor(
        [
            -1 if marked(tuple((basis >> bit) & 1 for bit in range(width))) else 1
            for basis in range(size)
        ],
        dtype=torch.complex128,
    )
    reflection = torch.full((size, size), 2 / size, dtype=torch.complex128) - torch.eye(
        size, dtype=torch.complex128
    )
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    for _ in range(iterations):
        circuit.any(*range(width), matrix=torch.diag(diagonal))
        circuit.any(*range(width), matrix=reflection)
    return circuit


def run_example() -> None:
    for iterations in (0, 1, 2):
        compare_states(build_program(iterations), reference_circuit(iterations), ("q",))


if __name__ == "__main__":
    run_example()
