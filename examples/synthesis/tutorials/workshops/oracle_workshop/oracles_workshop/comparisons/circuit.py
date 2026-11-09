"""Independent dense-oracle and reflection comparison for the workshop search."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.oracle_workshop.oracles_workshop.main import (
    build_program,
)


def reference_circuit(iterations: int = 2) -> fq.Circuit:
    width = 4
    size = 1 << width
    diagonal = torch.tensor(
        [-1 if basis // 4 - basis % 4 == 2 else 1 for basis in range(size)],
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
    compare_states(build_program(), reference_circuit(), ("x", "y"))


if __name__ == "__main__":
    run_example()
