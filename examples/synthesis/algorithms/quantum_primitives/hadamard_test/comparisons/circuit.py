"""Use the Fourier definition and inspect the entire entangled target state."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.hadamard_test.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qft.comparisons.circuit import fourier_matrix


def reference_circuit() -> fq.Circuit:
    controlled = torch.zeros((32, 32), dtype=torch.complex128)
    controlled[:16, :16] = torch.eye(16, dtype=torch.complex128)
    controlled[16:, 16:] = fourier_matrix(4)
    return (
        fq.Circuit(5, dtype=torch.complex128)
        .h(0)
        .any(*range(5), matrix=controlled)
        .h(0)
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("expectation_value", "psi"))


if __name__ == "__main__":
    run_example()
