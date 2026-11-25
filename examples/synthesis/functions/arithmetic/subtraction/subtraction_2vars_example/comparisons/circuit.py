"""Independent Circuit reference; no synthesis arithmetic or source translation."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.subtraction.subtraction_2vars_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    reference = fq.Circuit(10, dtype=torch.complex128)
    for wire in (0, 3, 5, 6, 7, 8, 9):
        reference.x(wire)
    return reference


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "b", "res"))


if __name__ == "__main__":
    run_example()
