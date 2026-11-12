"""Independent gate-level feature overlap in reversed inverse-layer order."""

import math

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.credit_card_fraud.main import (
    DATA_PAIRS,
    build_program,
)
from examples.synthesis.compare_states import compare_states

DTYPE = torch.complex128


def reference_circuit(data1: tuple[float, ...], data2: tuple[float, ...]) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=DTYPE)
    pairs = ((0, 1), (0, 2), (1, 2))
    for _ in range(2):
        for bit in range(3):
            circuit.h(2 - bit)
        for bit, value in enumerate(data1):
            circuit.rz(2 - bit, theta=2 * value)
        for first, second in pairs:
            circuit.rzz(
                2 - first,
                2 - second,
                theta=2 * (data1[first] - math.pi) * (data1[second] - math.pi),
            )
    # The adjoint reverses each E H layer, not only its rotation signs.
    for _ in range(2):
        for first, second in reversed(pairs):
            circuit.rzz(
                2 - first,
                2 - second,
                theta=-2 * (data2[first] - math.pi) * (data2[second] - math.pi),
            )
        for bit in reversed(range(3)):
            circuit.rz(2 - bit, theta=-2 * data2[bit])
        for bit in reversed(range(3)):
            circuit.h(2 - bit)
    return circuit


def run_example() -> None:
    for data1, data2 in DATA_PAIRS:
        compare_states(
            build_program(data1, data2), reference_circuit(data1, data2), ("qba",)
        )


if __name__ == "__main__":
    run_example()
