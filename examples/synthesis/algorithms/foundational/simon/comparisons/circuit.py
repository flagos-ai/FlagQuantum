"""Independent reversible representative table and explicit Hadamard Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.simon.main import build_program
from examples.synthesis.compare_states import compare_states


def value(x: int) -> int:
    return x if x < (x ^ 6) else x ^ 6


def reference_circuit() -> fq.Circuit:
    matrix = torch.zeros((1024, 1024), dtype=torch.complex128)
    for x in range(32):
        for y in range(32):
            matrix[(x << 5) | (y ^ value(x)), (x << 5) | y] = 1
    circuit = fq.Circuit(10, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    circuit.any(*range(10), matrix=matrix)
    for wire in range(5):
        circuit.h(wire)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "res"))


if __name__ == "__main__":
    run_example()
