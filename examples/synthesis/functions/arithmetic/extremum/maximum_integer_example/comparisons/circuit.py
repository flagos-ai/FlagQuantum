"""Independent Circuit reference; no synthesis arithmetic or source translation."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.extremum.maximum_integer_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    reference = fq.Circuit(7, dtype=torch.complex128)
    reference.h(0).h(1).h(2)
    for wire in (4, 5, 6):
        reference.x(wire).cx(0, wire)
    reference.cx(0, 3).ccx(0, 1, 4).ccx(0, 2, 5)
    return reference


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "res"))


if __name__ == "__main__":
    run_example()
