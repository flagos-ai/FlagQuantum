"""Reference the two conditional cyclic shifts as an exact basis permutation."""

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_1d.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    position: int = 0, selector: int = 0, coherent: bool = False
) -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(5):
        if position & (1 << (4 - wire)):
            circuit.x(wire)
    for bit in range(2):
        if selector & (1 << bit):
            circuit.x(6 - bit)
    if coherent:
        for wire in range(5):
            circuit.h(wire).phase(wire, theta=0.13 * (1 << (4 - wire)))
    circuit.h(5).h(6).z(5).z(6)
    shift = torch.zeros((128, 128), dtype=torch.complex128)
    for position in range(32):
        for selector in range(4):
            low, high = selector & 1, selector >> 1
            target = (position - (1 - low) + high) % 32
            shift[4 * target + selector, 4 * position + selector] = 1
    circuit.any(*range(7), matrix=shift)
    return circuit.h(5).h(6).z(5).z(6)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("position", "selector"))


if __name__ == "__main__":
    run_example()
