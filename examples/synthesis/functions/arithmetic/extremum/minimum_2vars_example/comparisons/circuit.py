"""Compare coherent minimum with an independently enumerated reversible table."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    # Each basis column maps (a,b,res) to (a,b,res XOR min(a,b)).
    columns = torch.arange(512)
    a = columns >> 6
    b = (columns >> 3) & 7
    rows = columns ^ torch.minimum(a, b)
    matrix = torch.zeros((512, 512), dtype=torch.complex128)
    matrix[rows, columns] = 1
    reference = fq.Circuit(9, dtype=torch.complex128).x(0)
    for wire in range(3, 6):
        reference.h(wire)
    return reference.any(*range(9), matrix=matrix)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "b", "res"))


if __name__ == "__main__":
    run_example()
