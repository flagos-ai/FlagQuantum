"""Independent Circuit reference; no synthesis arithmetic or source translation."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.negation.negation_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    reference = fq.Circuit(7, dtype=torch.complex128)
    matrix = torch.zeros((128, 128), dtype=torch.complex128)
    for a in range(8):
        encoded = (-(a if a < 4 else a - 8)) & 15
        for target in range(16):
            before = (a << 4) | target
            matrix[(a << 4) | (target ^ encoded), before] = 1
    for wire in range(3):
        reference.h(wire)
    reference.any(*range(7), matrix=matrix)
    return reference


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "b"))


if __name__ == "__main__":
    run_example()
