"""Independent RX/RZ Circuit with reversed inverse order and signed angles."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.qsvm.main import build_program
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    data1: tuple[float, float] = (0.23, -0.41),
    data2: tuple[float, float] = (-0.17, 0.37),
) -> fq.Circuit:
    return (
        fq.Circuit(1, dtype=torch.complex128)
        .rx(0, theta=math.pi * data1[0])
        .rz(0, theta=math.pi * data1[1])
        .rz(0, theta=-math.pi * data2[1])
        .rx(0, theta=-math.pi * data2[0])
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("qba",))


if __name__ == "__main__":
    run_example()
