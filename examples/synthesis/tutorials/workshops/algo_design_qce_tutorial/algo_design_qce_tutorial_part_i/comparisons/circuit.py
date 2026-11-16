"""Independent modular-addition permutation, without Fourier construction."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_i.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    permutation = torch.zeros((128, 128), dtype=torch.complex128)
    for x in range(8):
        for y in range(16):
            permutation[16 * x + (y + x * x) % 16, 16 * x + y] = 1
    return (
        fq.Circuit(7, dtype=torch.complex128)
        .h(0)
        .h(1)
        .h(2)
        .any(*range(7), matrix=permutation)
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "y"))


if __name__ == "__main__":
    run_example()
