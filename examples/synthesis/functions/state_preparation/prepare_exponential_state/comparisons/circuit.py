"""Independent exponential amplitudes embedded in a Householder Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.state_preparation.prepare_exponential_state.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    amplitudes = torch.softmax(
        -0.1 * torch.arange(32, dtype=torch.float64), dim=0
    ).sqrt()
    identity = torch.eye(32, dtype=torch.complex128)
    delta = identity[:, 0] - amplitudes
    reflection = identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    circuit = fq.Circuit(5, dtype=torch.complex128)
    circuit.any(*range(5), matrix=reflection)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
