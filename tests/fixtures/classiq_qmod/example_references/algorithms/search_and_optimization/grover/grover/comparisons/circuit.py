"""Validate the frozen >=4 Grover source against its dense reference."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

QMOD = Path(__file__).with_name("source.qmod")
EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))


def reference_circuit(iterations: int = 2) -> fq.Circuit:
    marked = {
        basis
        for basis in range(32)
        if sum(((basis >> left) ^ (basis >> right)) & 1 for left, right in EDGES) >= 4
    }
    diagonal = torch.ones(32, dtype=torch.complex128)
    diagonal[list(marked)] = -1
    reflection = torch.full((32, 32), 2 / 32, dtype=torch.complex128) - torch.eye(
        32, dtype=torch.complex128
    )
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    for _ in range(iterations):
        circuit.any(*range(5), matrix=torch.diag(diagonal))
        circuit.any(*range(5), matrix=reflection)
    return circuit


def run_example() -> None:
    for iterations in (0, 1, 2):
        program = synth.synthesize(
            QMOD, source_format="qmod", classical_inputs={"r": iterations}
        )
        compare_states(program, reference_circuit(iterations), ("nodes",))


if __name__ == "__main__":
    run_example()
