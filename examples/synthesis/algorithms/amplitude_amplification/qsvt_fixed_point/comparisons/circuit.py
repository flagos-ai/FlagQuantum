"""Independent seven-qubit projected-rotation matrices, with absolute phases."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.main import (
    PHASES,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def rotation_circuit(angle: float, initial: bool = False) -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    if initial:
        for wire in range(6):
            circuit.h(wire)
    values = []
    for word in range(64):
        a, b, c = word >> 4, (word >> 3) & 1, word & 7
        marked = word == 0 if initial else a + b == 3 and a - c == -2
        for aux in range(2):
            values.append(cmath.exp(1j * angle * ((aux ^ int(marked)) - 0.5)))
    circuit.any(
        *range(7), matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
    )
    if initial:
        for wire in range(6):
            circuit.h(wire)
    return circuit


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(7):
        circuit.h(wire)
    for index, angle in enumerate(phases):
        circuit.compose(rotation_circuit(angle, initial=index % 2 == 0))
    return circuit.h(6)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("a", "b", "c", "aux"))


if __name__ == "__main__":
    run_example()
