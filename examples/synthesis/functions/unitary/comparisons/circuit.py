"""Compare native gate lowering with direct independent matrix application."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.unitary.main import MATRIX, build_program


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128)
    matrix = torch.tensor(MATRIX, dtype=torch.complex128)
    for _ in range(1):
        circuit.any(0, 1, matrix=matrix)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
