"""Compare native gate lowering with direct independent matrix application."""

import torch

import flagquantum as fq
from examples.synthesis.applications.benchmarking.quantum_volume.main import (
    MATRIX,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    matrix = torch.tensor(MATRIX, dtype=torch.complex128)
    for _ in range(3):
        circuit.any(0, 1, matrix=matrix)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("target",))


if __name__ == "__main__":
    run_example()
