"""Independent preparation matrices, modular permutations and all32 RX entries."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding.tridiagonal.main import (
    AMPLITUDES,
    ANGLES,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def rotation(half_angle: float) -> torch.Tensor:
    c, s = math.cos(half_angle), math.sin(half_angle)
    return torch.tensor(((c, -s), (s, c)), dtype=torch.complex128)


def preparation(amplitudes: tuple[float, ...]) -> torch.Tensor:
    a, b, c, d = amplitudes
    root = rotation(math.atan2(math.hypot(c, d), math.hypot(a, b)))
    leaves = torch.zeros((4, 4), dtype=torch.complex128)
    leaves[:2, :2] = rotation(math.atan2(b, a))
    leaves[2:, 2:] = rotation(math.atan2(d, c))
    return leaves @ torch.kron(root, torch.eye(2, dtype=torch.complex128))


def reference_circuit(
    angles: tuple[float, ...] = ANGLES, amplitudes: tuple[float, ...] = AMPLITUDES
) -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(2, 6):
        circuit.h(wire)
    prep = preparation(amplitudes)
    circuit.any(0, 1, matrix=prep)
    decrement = torch.zeros((128, 128), dtype=torch.complex128)
    increment = torch.zeros((128, 128), dtype=torch.complex128)
    table = torch.zeros((128, 128), dtype=torch.complex128)
    for selector in range(4):
        for position in range(16):
            base = 32 * selector + 2 * position
            half = angles[2 * position + (selector & 1)] / 2
            c, s = math.cos(half), math.sin(half)
            table[base, base] = table[base + 1, base + 1] = c
            table[base, base + 1] = table[base + 1, base] = torch.tensor(
                -1j * s, dtype=torch.complex128
            )
            for data in range(2):
                decrement[
                    32 * selector + 2 * ((position - int(selector == 1)) % 16) + data,
                    base + data,
                ] = 1
                increment[
                    32 * selector + 2 * ((position + int(selector >= 2)) % 16) + data,
                    base + data,
                ] = 1
    circuit.any(*range(7), matrix=decrement)
    circuit.any(*range(7), matrix=table)
    circuit.any(*range(7), matrix=increment)
    return circuit.any(0, 1, matrix=prep.conj().T)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("s", "j", "data"))


if __name__ == "__main__":
    run_example()
