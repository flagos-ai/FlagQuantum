"""Compare native arithmetic with an independently constructed Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.comparator.comparator_integer_example.main import (
    main,
)
from flagquantum.synth import synthesize


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(3):
        circuit.h(wire)
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    for index in range(16):
        a, flag = divmod(index, 2)
        matrix[(a << 1) | (flag ^ int(a <= 2)), index] = 1
    return circuit.any(0, 1, 2, 3, matrix=matrix)


def run_example() -> None:
    compare_states(synthesize(main), reference_circuit(), ("a", "res"))


if __name__ == "__main__":
    run_example()
