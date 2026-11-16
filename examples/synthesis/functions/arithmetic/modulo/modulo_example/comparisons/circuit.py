"""Independently prepare the classical source inputs and calculated result."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.modulo.modulo_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    reference = fq.Circuit(12, dtype=torch.complex128)
    for wire in (2, 7, 8, 9, 10, 11):
        reference.x(wire)
    return reference


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "b", "res"))


if __name__ == "__main__":
    run_example()
