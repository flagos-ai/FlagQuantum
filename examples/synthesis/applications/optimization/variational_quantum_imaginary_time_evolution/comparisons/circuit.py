"""Independent dense Pauli exponentials for all six declared generators."""

import math

import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states

WORDS = ("IYZI", "IIZY", "YIZI", "IZIY", "YZII", "YIIZ")


def reference_circuit(
    angles: tuple[float, ...] = (0.13, -0.21, 0.32, -0.17, 0.29, -0.41),
) -> fq.Circuit:
    dtype = torch.complex128
    axes = {
        "I": torch.eye(2, dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.diag(torch.tensor([1, -1], dtype=dtype)),
    }
    circuit = fq.Circuit(4, dtype=dtype)
    for wire in range(4):
        circuit.h(wire)
    for word, angle in zip(WORDS, angles, strict=True):
        matrix = torch.ones((1, 1), dtype=dtype)
        for axis in reversed(word):
            matrix = torch.kron(matrix, axes[axis])
        exponential = (
            math.cos(angle) * torch.eye(16, dtype=dtype) - 1j * math.sin(angle) * matrix
        )
        circuit.any(0, 1, 2, 3, matrix=exponential)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("nodes",))


if __name__ == "__main__":
    run_example()
