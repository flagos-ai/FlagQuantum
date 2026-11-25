"""Verify angle encoding with independent RY Circuit operations."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.data_encoding.encode_in_angle.main import (
    build_program,
)


def reference_circuit(data: Sequence[float]) -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire, value in enumerate(reversed(data)):
        circuit.ry(wire, theta=math.pi * value)
    return circuit


def run_example() -> None:
    for data in ([0.2, -0.3, 0.4, 0.5], [0.17, 2.63, -2.49, 0.91]):
        compare_states(build_program(data), reference_circuit(data), ("x",))


if __name__ == "__main__":
    run_example()
