"""Check Bell-state addition against an independently enumerated permutation."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.add_bell_states.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).x(3)
    matrix = torch.zeros((128, 128), dtype=torch.complex128)
    for value in range(128):
        a = value >> 5
        b = (value >> 3) & 3
        matrix[value ^ (a + b), value] = 1
    return circuit.any(*range(7), matrix=matrix)


def run_example() -> None:
    compare_states(synth.synthesize(main), reference_circuit(), ("a", "b", "total"))


if __name__ == "__main__":
    run_example()
