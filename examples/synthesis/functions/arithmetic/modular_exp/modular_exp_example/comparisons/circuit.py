"""Apply an independent classical modular permutation to the quantum inputs."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.modular_exp.modular_exp_example.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    matrix = torch.zeros((64, 64), dtype=torch.complex128)
    for column in range(64):
        power, x = divmod(column, 8)
        target = (pow(4, power, 5) * x) % 5 if x < 5 else x
        matrix[8 * power + target, column] = 1
    reference = fq.Circuit(6, dtype=torch.complex128).x(5)
    for wire in range(3):
        reference.h(wire)
    return reference.any(*range(6), matrix=matrix)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("power", "x"))


if __name__ == "__main__":
    run_example()
