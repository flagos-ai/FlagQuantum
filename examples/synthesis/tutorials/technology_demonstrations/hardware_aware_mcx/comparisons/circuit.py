"""Independent basis oracle for the retained all-zero input program."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.hardware_aware_mcx.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    # Every original control is zero; the multi-control X acts as identity here.
    return fq.Circuit(16, dtype=torch.complex128)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("cntrl", "target"))


if __name__ == "__main__":
    run_example()
