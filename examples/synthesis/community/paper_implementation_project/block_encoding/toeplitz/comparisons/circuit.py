"""Independent direct RX blocks and signed classical snapshot permutation."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding.toeplitz.main import (
    ANGLES,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(angles: tuple[float, ...] = ANGLES) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for wire in range(1, 7):
        circuit.h(wire)
    blocks = []
    for angle in angles:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        blocks.append(
            torch.tensor(((c, -1j * s), (-1j * s, c)), dtype=torch.complex128)
        )
    table = torch.zeros((8, 8), dtype=torch.complex128)
    for index, block in enumerate(blocks):
        table[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = block
    circuit.any(1, 2, 0, matrix=table)
    permutation = torch.zeros((128, 128), dtype=torch.complex128)
    for selector in range(4):
        for position in range(16):
            signed_snapshot = position + selector - 1
            for flag in range(2):
                before = 32 * selector + 2 * position + flag
                after = (
                    32 * selector
                    + 2 * (signed_snapshot % 16)
                    + (flag ^ int(not 0 <= signed_snapshot < 16))
                )
                permutation[after, before] = 1
    circuit.any(*range(1, 8), matrix=permutation)
    return circuit.h(1).h(2)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("data", "s", "j", "dlt"))


if __name__ == "__main__":
    run_example()
