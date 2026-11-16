"""Independent Boolean truth-table Circuit for the complete three-bit domain."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.arithmetic_expressions.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.h(0).h(1).h(2)
    # z is true for x>=2, or x==1 and y==1.
    matrix = torch.eye(16, dtype=torch.complex128)
    for x in range(4):
        for y in range(2):
            if x >= 2 or (x == 1 and y == 1):
                first = 4 * x + 2 * y
                matrix[first, first] = matrix[first + 1, first + 1] = 0
                matrix[first, first + 1] = matrix[first + 1, first] = 1
    circuit.any(0, 1, 2, 3, matrix=matrix)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "y", "z"))


if __name__ == "__main__":
    run_example()
