"""Independent per-signal Circuit with explicit logical-to-physical bit order."""

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    message: tuple[int, ...] = (0, 0, 1, 1, 0, 0, 0, 1),
    alice_bases: tuple[int, ...] = (1, 1, 0, 1, 1, 1, 0, 1),
    bob_bases: tuple[int, ...] = (1, 0, 0, 0, 0, 1, 1, 0),
) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for logical, (bit, alice, bob) in enumerate(
        zip(message, alice_bases, bob_bases, strict=True)
    ):
        wire = 7 - logical
        if bit:
            circuit.x(wire)
        if alice:
            circuit.h(wire)
        if bob:
            circuit.h(wire)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("qba",))


if __name__ == "__main__":
    run_example()
