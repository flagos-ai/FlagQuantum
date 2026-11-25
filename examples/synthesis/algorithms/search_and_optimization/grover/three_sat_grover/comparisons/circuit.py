"""Independent Boolean marking and phase-exact diffusion matrices."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states

CLAUSES = (
    ((1, 1), (2, 1), (3, 1)),
    ((0, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (2, 1)),
    ((0, 1), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (2, 1)),
    ((0, 0), (1, 0), (3, 0)),
    ((0, 0), (1, 0), (3, 1)),
    ((1, 0), (2, 0), (3, 0)),
    ((1, 1), (2, 0), (3, 1)),
    ((0, 1), (2, 0), (3, 1)),
    ((0, 1), (1, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0)),
)


def marked(bits: tuple[int, ...]) -> bool:
    return all(
        any(bits[index] == value for index, value in clause) for clause in CLAUSES
    )


def reference_circuit(r: int = 2) -> fq.Circuit:
    width = 4
    size = 1 << width
    phases = torch.tensor(
        [
            -1.0 if marked(tuple((basis >> bit) & 1 for bit in range(width))) else 1.0
            for basis in range(size)
        ],
        dtype=torch.complex128,
    )
    oracle = torch.diag(phases)
    diffusion = torch.full((size, size), 2 / size, dtype=torch.complex128) - torch.eye(
        size, dtype=torch.complex128
    )
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    for _ in range(r):
        circuit.any(*range(width), matrix=oracle)
        circuit.any(*range(width), matrix=diffusion)
    return circuit


def run_example() -> None:
    for r in (0, 1, 2):
        compare_states(build_program(r), reference_circuit(r), ("q",))


if __name__ == "__main__":
    run_example()
