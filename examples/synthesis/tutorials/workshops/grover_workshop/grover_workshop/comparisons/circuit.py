"""Compare the native arithmetic oracle with an independent dense reflection."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    width = 7
    size = 1 << width
    diagonal = torch.tensor(
        [
            -1 if (basis & 7) == 2 * (basis >> 5) + ((basis >> 3) & 3) else 1
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
    circuit.any(*range(width), matrix=torch.diag(diagonal))
    circuit.any(*range(width), matrix=reflection)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "b", "c"))


if __name__ == "__main__":
    run_example()
