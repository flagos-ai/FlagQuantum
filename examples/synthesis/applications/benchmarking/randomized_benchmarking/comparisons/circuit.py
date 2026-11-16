"""Independent direct Circuit SX sequence and reverse gate inverse."""

import math

import torch

import flagquantum as fq
from examples.synthesis.applications.benchmarking.randomized_benchmarking.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states

SEQUENCE = (
    "RZ",
    "X",
    "SX",
    "I",
    "I",
    "SX",
    "I",
    "RZ",
    "SX",
    "RZ",
    "X",
    "X",
    "I",
    "X",
    "I",
    "SX",
    "RZ",
    "X",
    "RZ",
    "X",
    "X",
    "X",
    "RZ",
    "X",
    "I",
)


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(1, dtype=torch.complex128)
    for gate in SEQUENCE:
        if gate == "RZ":
            circuit.rz(0, theta=math.pi)
        else:
            circuit.gate(gate.lower(), 0)
    for gate in reversed(SEQUENCE):
        if gate == "RZ":
            circuit.rz(0, theta=-math.pi)
        else:
            circuit.gate("sxdg" if gate == "SX" else gate.lower(), 0)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("target",))


if __name__ == "__main__":
    run_example()
