"""Independent dense source permutations and analytical preparation matrices."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding.laplacian_4x4.main import (
    ANGLES,
    PROBABILITIES,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def preparation(probabilities: tuple[float, ...]) -> torch.Tensor:
    amplitudes = tuple(math.sqrt(value) for value in probabilities)
    unitary = torch.eye(8, dtype=torch.complex128)
    for stride in (4, 2, 1):
        stage = torch.zeros((8, 8), dtype=torch.complex128)
        for start in range(0, 8, 2 * stride):
            left = math.hypot(*amplitudes[start : start + stride])
            right = math.hypot(*amplitudes[start + stride : start + 2 * stride])
            angle = math.atan2(right, left)
            c, s = math.cos(angle), math.sin(angle)
            for tail in range(stride):
                lo, hi = start + tail, start + stride + tail
                stage[lo, lo] = stage[hi, hi] = c
                stage[lo, hi], stage[hi, lo] = -s, s
        unitary = stage @ unitary
    return unitary


def block_matrix(angles: tuple[float, ...]) -> torch.Tensor:
    matrix = torch.zeros((512, 512), dtype=torch.complex128)
    for selector in range(8):
        s0, s1, s2 = selector & 1, (selector >> 1) & 1, selector >> 2
        for position in range(16):
            shifted = (position + 4 * ((1 - s0) * s2)) % 16
            shifted = (shifted + (1 - s0) * s1) % 16
            mark = (
                (s0 * (1 - s1) * (1 - s2))
                ^ (s1 * int(shifted % 4 == 0))
                ^ (s2 * int(shifted // 4 == 0))
            )
            final_s0 = s0 ^ s1 ^ s2
            final_selector = selector ^ s1 ^ s2
            final_position = (shifted - (1 - final_s0) * s1) % 16
            final_position = (final_position - 4 * (1 - final_s0) * s2) % 16
            angle = angles[selector >> 1] if selector >> 1 < 3 else 0.0
            c, s = math.cos(angle / 2), math.sin(angle / 2)
            rx = ((complex(c), -1j * s), (-1j * s, complex(c)))
            for data in range(2):
                for flag in range(2):
                    column = 64 * selector + 4 * position + 2 * data + flag
                    for output_data in range(2):
                        row = (
                            64 * final_selector
                            + 4 * final_position
                            + 2 * output_data
                            + (flag ^ mark)
                        )
                        matrix[row, column] = torch.tensor(
                            (-1) ** data * rx[output_data][data], dtype=torch.complex128
                        )
    return matrix


def reference_circuit(
    angles: tuple[float, ...] = ANGLES,
    probabilities: tuple[float, ...] = PROBABILITIES,
    initial_data: int = 0,
    initial_flag: int = 0,
) -> fq.Circuit:
    circuit = fq.Circuit(9, dtype=torch.complex128)
    if initial_data:
        circuit.x(7)
    if initial_flag:
        circuit.x(8)
    for wire in range(3, 7):
        circuit.h(wire)
    prep = preparation(probabilities)
    circuit.any(0, 1, 2, matrix=prep)
    circuit.any(*range(9), matrix=block_matrix(angles))
    return circuit.any(0, 1, 2, matrix=prep.conj().T)


def run_example() -> None:
    compare_states(
        build_program(), reference_circuit(), ("s", "j1", "j0", "data", "dlt")
    )


if __name__ == "__main__":
    run_example()
