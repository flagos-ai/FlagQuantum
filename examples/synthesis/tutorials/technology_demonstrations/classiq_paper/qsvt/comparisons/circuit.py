"""Independent block matrices, permutations and analytic projected phases."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.main import (
    PHASES,
)

DTYPE = torch.complex128


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    # Qmod arrays are little-endian within each named output register.
    projection, deletion, select, aux = 8, 0, 7, 9
    data = tuple(range(1, 7))
    circuit = fq.Circuit(10, dtype=DTYPE)
    shift = torch.zeros((256, 256), dtype=DTYPE)
    for selector in range(2):
        for word in range(128):
            displacement = 1 if selector == 0 else -1
            shift[
                selector * 128 + (word + displacement) % 128, selector * 128 + word
            ] = 1
    reflection = torch.diag(
        torch.tensor(
            [1 if (column & 64) or not (column & 63) else -1 for column in range(128)],
            dtype=DTYPE,
        )
    )

    def project(angle: float) -> None:
        diagonal = [
            cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
            for word in range(16)
        ]
        circuit.any(
            projection,
            deletion,
            select,
            aux,
            matrix=torch.diag(torch.tensor(diagonal, dtype=DTYPE)),
        )

    def reflect() -> None:
        circuit.h(projection)
        circuit.any(projection, *data, matrix=reflection)
        circuit.h(projection)

    def block_encoding(inverse: bool = False) -> None:
        if inverse:
            reflect()
        circuit.h(select)
        circuit.any(
            select, deletion, *data, matrix=shift.mH.contiguous() if inverse else shift
        )
        circuit.h(select)
        if not inverse:
            reflect()

    circuit.h(aux)
    project(phases[0])
    block_encoding()
    project(phases[1])
    block_encoding(inverse=True)
    project(phases[2])
    block_encoding()
    project(phases[3])
    circuit.h(aux)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    expected = fq.run(reference_circuit()).statevector().flatten()
    actual = fq.run(program).statevector().reshape(expected.numel(), -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
