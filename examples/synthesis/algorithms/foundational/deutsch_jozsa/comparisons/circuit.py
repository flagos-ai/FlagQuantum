"""Use the original predicates' truth tables as independent phase oracles."""

from collections.abc import Callable

import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.deutsch_jozsa.main import main, simple
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def circuit_reference(width: int, predicate: Callable[[int], bool]) -> fq.Circuit:
    signs = [(-1 if predicate(value) else 1) for value in range(1 << width)]
    oracle = torch.diag(torch.tensor(signs, dtype=torch.complex128))
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    circuit.any(*range(width), matrix=oracle)
    for wire in range(width):
        circuit.h(wire)
    return circuit


def complex_predicate(value: int) -> bool:
    return max(value ^ 4, value & 3) >= 4


def run_example() -> None:
    compare_states(
        synthesize(simple, max_qubits=11),
        circuit_reference(4, lambda value: value > 7),
        ("x",),
    )
    compare_states(
        synthesize(main, max_qubits=11),
        circuit_reference(3, complex_predicate),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
