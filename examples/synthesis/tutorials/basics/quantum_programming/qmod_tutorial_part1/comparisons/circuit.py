"""Independent retained-program Circuit reference."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    # Independent diagonal rule: flip all-ones and words with bit3 set.
    angles = torch.tensor(
        [torch.pi * ((basis == 15) + ((basis >> 3) & 1)) for basis in range(16)],
        dtype=torch.float64,
    )
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    circuit.any(0, 1, 2, 3, matrix=torch.diag(torch.exp(1j * angles)))
    for wire in range(4):
        circuit.h(wire)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
