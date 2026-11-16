"""Independent coordinate-shift permutation in original packed-register order."""

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    position: int = 0, selector: int = 0, dimension: int = 0, coherent: bool = False
) -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128)
    word = 8 * position + 2 * selector + dimension
    for wire in range(6):
        if word & (1 << (5 - wire)):
            circuit.x(wire)
    if coherent:
        for wire, angle in enumerate((0.26, 0.13, 0.39, 0.52, 0.65, 0.78)):
            circuit.h(wire).phase(wire, theta=angle)
    circuit.h(3).h(4).z(3).z(4).h(5)
    shift = torch.zeros((64, 64), dtype=torch.complex128)
    for position in range(8):
        for selector in range(4):
            for dimension in range(2):
                low, high = position & 1, position >> 1
                delta = -int(not (selector & 1)) + int(bool(selector & 2))
                if dimension:
                    high = (high + delta) % 4
                else:
                    low = (low + delta) % 2
                target = 2 * high + low
                shift[
                    8 * target + 2 * selector + dimension,
                    8 * position + 2 * selector + dimension,
                ] = 1
    circuit.any(*range(6), matrix=shift)
    return circuit.h(3).h(4).h(5)


def run_example() -> None:
    compare_states(
        build_program(), reference_circuit(), ("j1", "j0", "selector", "dimension")
    )


if __name__ == "__main__":
    run_example()
