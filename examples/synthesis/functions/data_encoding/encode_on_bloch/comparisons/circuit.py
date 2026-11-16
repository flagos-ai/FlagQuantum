"""Verify paired Bloch angles and the odd tail with an independent Circuit."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.data_encoding.encode_on_bloch.main import (
    build_program,
)


def reference_circuit(data: Sequence[float]) -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for pair in range(3):
        wire = 3 - pair
        circuit.rx(wire, theta=math.pi * data[2 * pair])
        circuit.rz(wire, theta=math.pi * data[2 * pair + 1])
    circuit.rx(0, theta=2 * math.pi * data[6])
    return circuit


def run_example() -> None:
    for data in (
        [0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
        [0.17, 2.63, -2.49, 0.91, -0.25, 0.37, 1.58],
    ):
        compare_states(build_program(data), reference_circuit(data), ("x",))


if __name__ == "__main__":
    run_example()
