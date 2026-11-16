"""Independent Circuit reference; no synthesis arithmetic or source translation."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.multiplication.multiplication_float_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    reference = fq.Circuit(7, dtype=torch.complex128)
    matrix = torch.zeros((128, 128), dtype=torch.complex128)
    for a in range(4):
        encoded = (7 * a) & 31
        for target in range(32):
            before = (a << 5) | target
            matrix[(a << 5) | (target ^ encoded), before] = 1
    for wire in range(2):
        reference.h(wire)
    reference.any(*range(7), matrix=matrix)
    return reference


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "res"))


if __name__ == "__main__":
    run_example()
