"""Independent controlled-RX table and packed-field permutations."""

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding.checkerboard.main import (
    ANGLES,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    angles: tuple[float, float] = ANGLES, word: int = 0, coherent: bool = False
) -> fq.Circuit:
    c = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(7):
        if word & (1 << (6 - wire)):
            c.x(wire)
        if coherent:
            c.h(wire).phase(wire, theta=0.13 * (wire + 1))
    for q in (4, 5, 6, 1, 2, 3):
        c.h(q)
    c.swap(2, 6).swap(1, 5).cx(3, 4)
    # Direct independently enumerated controlled-RX table.
    rx = []
    for a in angles:
        v = torch.tensor(a, dtype=torch.float64)
        rx.append(
            torch.stack(
                (
                    torch.stack((torch.cos(v / 2), -1j * torch.sin(v / 2))),
                    torch.stack((-1j * torch.sin(v / 2), torch.cos(v / 2))),
                )
            )
        )
    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    matrix[:2, :2] = rx[0]
    matrix[2:, 2:] = rx[1]
    c.any(3, 0, matrix=matrix)
    for q in (1, 2, 3):
        c.h(q)
    return c


def run_example() -> None:
    compare_states(
        build_program(), reference_circuit(), ("data", "s1", "s0", "j1", "j0")
    )


if __name__ == "__main__":
    run_example()
