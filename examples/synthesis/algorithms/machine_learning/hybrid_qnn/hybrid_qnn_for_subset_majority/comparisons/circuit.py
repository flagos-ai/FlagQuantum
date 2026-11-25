"""Compare the entire variational layer against independent RX/RZ/RY/RZZ gates."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(data: Sequence[float], weights: Sequence[float]) -> fq.Circuit:
    reference = fq.Circuit(2, dtype=torch.complex128)
    reference.rx(1, theta=math.pi * data[0]).rz(1, theta=math.pi * data[1])
    reference.rx(0, theta=math.pi * data[2]).rz(0, theta=math.pi * data[3])
    reference.ry(1, theta=weights[0]).ry(0, theta=weights[1])
    reference.rzz(1, 0, theta=weights[2])
    return reference


def run_example() -> None:
    for scale in (1.0, -0.63):
        data = [0.2 * scale, -0.3 * scale, 0.4 * scale, -0.5 * scale]
        weights = [0.1 * scale * (i + 1) for i in range(3)]
        compare_states(
            build_program(data, weights), reference_circuit(data, weights), ("result",)
        )


if __name__ == "__main__":
    run_example()
