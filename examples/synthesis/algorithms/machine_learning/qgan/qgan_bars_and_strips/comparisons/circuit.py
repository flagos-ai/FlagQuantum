"""Compare the entire variational layer against independent RX/RZ/RY/RZZ gates."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(data: Sequence[float], weights: Sequence[float]) -> fq.Circuit:
    reference = fq.Circuit(4, dtype=torch.complex128)
    for index, wire in enumerate((3, 2, 1, 0)):
        reference.rx(wire, theta=math.pi * data[index])
        reference.rz(wire, theta=math.pi * data[index])
        reference.ry(wire, theta=weights[index])
    reference.rzz(3, 2, theta=weights[4])
    reference.rzz(2, 1, theta=weights[5])
    reference.rzz(1, 0, theta=weights[6])
    reference.rzz(3, 2, theta=weights[7])
    return reference


def run_example() -> None:
    for scale in (1.0, -0.63):
        data = [0.2 * scale, -0.3 * scale, 0.4 * scale, -0.5 * scale]
        weights = [0.1 * scale * (i + 1) for i in range(8)]
        compare_states(
            build_program(data, weights), reference_circuit(data, weights), ("result",)
        )


if __name__ == "__main__":
    run_example()
