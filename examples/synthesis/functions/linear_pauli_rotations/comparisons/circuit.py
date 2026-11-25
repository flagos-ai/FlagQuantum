"""Use an independent block matrix for the complete affine-angle operation."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.linear_pauli_rotations.main import build_program


def rotation(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.complex128)


def reference_circuit(
    slopes: Sequence[float] = (2.1, 1.0, 7.0),
    offsets: Sequence[float] = (0.1, 0.3, 0.33),
    *,
    input_value: int = 0,
    coherent: bool = False,
) -> fq.Circuit:
    circuit = fq.Circuit(9, dtype=torch.complex128)
    for bit in range(6):
        if (input_value >> bit) & 1:
            circuit.x(5 - bit)
    if coherent:
        for wire in range(6):
            circuit.h(wire)
        circuit.z(5)
    matrix = torch.zeros((512, 512), dtype=torch.complex128)
    for value in range(64):
        block = torch.ones((1, 1), dtype=torch.complex128)
        for target in reversed(range(3)):
            block = torch.kron(
                block, rotation(slopes[target] * value + offsets[target])
            )
        matrix[8 * value : 8 * (value + 1), 8 * value : 8 * (value + 1)] = block
    circuit.any(*range(9), matrix=matrix)
    return circuit


def run_example() -> None:
    for coherent in (False, True):
        compare_states(
            build_program(coherent=coherent),
            reference_circuit(coherent=coherent),
            ("x", "ind"),
        )


if __name__ == "__main__":
    run_example()
