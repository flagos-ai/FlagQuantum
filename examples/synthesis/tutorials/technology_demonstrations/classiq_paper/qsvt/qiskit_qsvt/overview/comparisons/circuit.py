"""Independent literal Fourier adder and block matrices; no synthesis helpers."""

import cmath
import math

import torch

import flagquantum as fq

from ..program import PHASES

DTYPE = torch.complex128


def source_fourier_adder(constant: int) -> torch.Tensor:
    """Compute all 128 columns of the source no-swaps QFT/P/inverse QFT."""
    width = 7
    size = 1 << width
    qft = torch.eye(size, dtype=DTYPE)
    for j in reversed(range(width)):
        mask = 1 << (width - 1 - j)
        gate = torch.zeros((size, size), dtype=DTYPE)
        for word in range(size):
            gate[word & ~mask, word] = 2**-0.5
            gate[word | mask, word] = (-1 if word & mask else 1) * 2**-0.5
        qft = gate @ qft
        for k in reversed(range(j)):
            other = 1 << (width - 1 - k)
            angle = math.pi / 2 ** (j - k)
            diagonal = torch.tensor(
                [
                    cmath.exp(1j * angle) if word & mask and word & other else 1
                    for word in range(size)
                ],
                dtype=DTYPE,
            )
            qft = diagonal[:, None] * qft
    diagonal = torch.ones(size, dtype=DTYPE)
    for j in range(width):
        angle = (constant % 2 ** (j + 1)) * math.pi / 2**j
        diagonal *= torch.tensor(
            [
                cmath.exp(1j * angle) if word & (1 << (width - 1 - j)) else 1
                for word in range(size)
            ],
            dtype=DTYPE,
        )
    return qft.mH @ (diagonal[:, None] * qft)


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    """Use canonical block/data/aux order and preserve absolute scalar phase.

    The largest dense operator is 256 by 256 complex128 (1 MiB), used only
    for local correctness inspection. Arithmetic is independent of lowering.
    """
    circuit = fq.Circuit(10, dtype=DTYPE)
    select, deletion, projection, aux = 2, 1, 0, 9
    data = tuple(range(3, 9))
    shift = torch.zeros((256, 256), dtype=DTYPE)
    for selector in range(2):
        for word in range(128):
            displacement = 1 if selector == 0 else -1
            shift[
                selector * 128 + (word + displacement) % 128, selector * 128 + word
            ] = 1
    reflection = torch.diag(
        torch.tensor([-1 if column == 0 else 1 for column in range(128)], dtype=DTYPE)
    )

    def project(angle: float) -> None:
        diagonal = [
            cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
            for word in range(16)
        ]
        circuit.any(
            select,
            deletion,
            projection,
            aux,
            matrix=torch.diag(torch.tensor(diagonal, dtype=DTYPE)),
        )

    def block(inverse: bool = False) -> None:
        circuit.h(select).h(projection)
        if inverse:
            circuit.any(projection, *data, matrix=reflection)
        circuit.any(
            select,
            deletion,
            *data,
            matrix=shift.mH.contiguous() if inverse else shift,
        )
        if not inverse:
            circuit.any(projection, *data, matrix=reflection)
        circuit.h(select).h(projection)

    circuit.h(aux)
    project(phases[0])
    block()
    project(phases[1])
    block(inverse=True)
    project(phases[2])
    block()
    project(phases[3])
    circuit.h(aux)
    return circuit


if __name__ == "__main__":
    from ..workflow import run_workflow

    workflow = run_workflow()
    print("Native and CX/U3 absolute states match the independent source Circuit.")
    print(workflow.resources)
