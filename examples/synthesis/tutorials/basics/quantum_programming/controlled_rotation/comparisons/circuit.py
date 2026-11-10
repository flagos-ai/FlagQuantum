"""Compare native gate composition with a dense controlled-rotation matrix."""

import math

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.controlled_rotation.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    matrix = torch.eye(8, dtype=torch.complex128)
    cosine, sine = math.cos(math.pi / 6), math.sin(math.pi / 6)
    matrix[2:4, 2:4] = torch.tensor(
        [[cosine, -sine], [sine, cosine]], dtype=torch.complex128
    )
    return (
        fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2).any(0, 1, 2, matrix=matrix)
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("msb", "lsb"))


if __name__ == "__main__":
    run_example()
