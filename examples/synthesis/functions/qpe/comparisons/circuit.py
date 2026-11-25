"""Compare diagonal QPE with independently known phase-label correlations."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qpe.main import build_program


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(4, dtype=torch.complex128).h(0).h(1).cx(0, 3).cx(1, 2)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("state", "phase"))


if __name__ == "__main__":
    run_example()
