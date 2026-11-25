"""Independent modular permutations; no Fourier gates or synthesis arithmetic."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.main import (
    build_program,
)


def reference_circuit(
    width: int = 10,
    steps: int = 1,
    balanced: bool = False,
    position: int = 0,
    coherent: bool = False,
) -> fq.Circuit:
    """Use exact basis permutations at small width and a closed one-step state."""
    circuit = fq.Circuit(width + 1, dtype=torch.complex128)
    if width > 7:
        if (steps, balanced, position, coherent) != (1, False, 0, False):
            raise ValueError("wide reference only admits the original one-step state")
        # Starting from zero: (|+1,0> + |-1,1>)/sqrt(2).
        circuit.h(width).x(width - 1)
        for wire in range(width - 1):
            circuit.cx(width, wire)
        return circuit
    for wire in range(width):
        if position & (1 << (width - 1 - wire)):
            circuit.x(wire)
    if coherent:
        for wire in range(width):
            circuit.h(wire)
        circuit.h(width).s(width)
    if balanced:
        circuit.h(width).s(width)
    size = 1 << width
    shift = torch.zeros((2 * size, 2 * size), dtype=torch.complex128)
    for x in range(size):
        for coin in range(2):
            y = (x + (1 if coin == 0 else -1)) % size
            shift[2 * y + coin, 2 * x + coin] = 1
    for _ in range(steps):
        circuit.h(width).any(*range(width + 1), matrix=shift)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "coin"))


if __name__ == "__main__":
    run_example()
