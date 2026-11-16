"""Check native compression against a Circuit with explicit two-level matrices."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(basis: int | None = None) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    if basis is None:
        circuit.h(0).h(1).h(2).y(2)
    else:
        for bit in range(3):
            if (basis >> bit) & 1:
                circuit.x(2 - bit)
    first = torch.eye(8, dtype=torch.complex128)
    first[2, 2], first[5, 5] = 1 / math.sqrt(3), -1 / math.sqrt(3)
    first[2, 5] = first[5, 2] = math.sqrt(2 / 3)
    second = torch.eye(8, dtype=torch.complex128)
    second[1, 1] = second[4, 4] = math.sqrt(2 / 3)
    second[1, 4], second[4, 1] = 1 / math.sqrt(3), -1 / math.sqrt(3)
    return (
        circuit.cx(1, 2)
        .ch(2, 1)
        .any(0, 1, 2, matrix=first)
        .any(0, 1, 2, matrix=second)
        .cx(0, 1)
        .cx(0, 1)
        .cx(0, 2)
        .ccx(2, 1, 0)
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
