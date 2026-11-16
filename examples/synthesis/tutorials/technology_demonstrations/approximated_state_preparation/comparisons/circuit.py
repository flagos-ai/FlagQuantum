"""Independently prepare the target with a real Householder reflection."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.main import (
    AMPLITUDES,
    build_program,
)


def reference_circuit() -> fq.Circuit:
    target = torch.tensor(AMPLITUDES, dtype=torch.complex128)
    target = target / torch.linalg.vector_norm(target)
    difference = -target.clone()
    difference[0] += 1
    reflection = torch.eye(64, dtype=torch.complex128) - 2 * torch.outer(
        difference, difference.conj()
    ) / torch.vdot(difference, difference)
    return fq.Circuit(6, dtype=torch.complex128).any(*range(6), matrix=reflection)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("out",))


if __name__ == "__main__":
    run_example()
