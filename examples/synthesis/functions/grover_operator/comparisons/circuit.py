"""Independent dense-oracle and reflection comparison for the search program."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.grover_operator.main import build_program


def reference_circuit() -> fq.Circuit:
    width = 4
    size = 1 << width
    marked = [
        x * 4 + y for x in range(4) for y in range(4) if x + y < 4 and (x * y) % 4 == 2
    ]
    oracle = torch.ones(size, dtype=torch.complex128)
    oracle[marked] = -1
    reflection = torch.full((size, size), 2 / size, dtype=torch.complex128) - torch.eye(
        size, dtype=torch.complex128
    )
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    for _ in range(2):
        circuit.any(*range(width), matrix=torch.diag(oracle))
        circuit.any(*range(width), matrix=reflection)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "y"))


if __name__ == "__main__":
    run_example()
