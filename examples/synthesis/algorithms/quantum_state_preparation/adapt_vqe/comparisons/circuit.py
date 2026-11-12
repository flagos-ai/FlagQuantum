"""Use a direct XX rotation as the independent ansatz reference."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_state_preparation.adapt_vqe.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(theta: float = 0.23) -> fq.Circuit:
    return fq.Circuit(4, dtype=torch.complex128).rxx(0, 3, theta=2 * theta)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("v",))


if __name__ == "__main__":
    run_example()
