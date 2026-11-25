"""Independent cut-count phase matrices and Circuit mixer rotations."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.main import (
    build_program,
)

BINDINGS = (
    (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    (-0.17, 0.29, -0.31, 0.43, -0.53, 0.61, -0.71, 0.83),
)


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
    cuts = torch.tensor(
        [
            sum(((basis >> a) & 1) != ((basis >> b) & 1) for a, b in edges)
            for basis in range(32)
        ],
        dtype=torch.float64,
    )
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    for layer in range(4):
        circuit.any(
            *range(5), matrix=torch.diag(torch.exp(-1j * params[2 * layer] * cuts))
        )
        for wire in range(5):
            circuit.rx(wire, theta=params[2 * layer + 1])
    return circuit


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
