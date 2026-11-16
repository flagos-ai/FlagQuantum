"""Independent matrix sequence with analytic projected phases and explicit bit order."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.data import (
    AMPLITUDES,
    MATRIX,
    PHASES,
)


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    rhs = torch.tensor(AMPLITUDES, dtype=torch.complex128)
    rhs = rhs / torch.linalg.vector_norm(rhs)
    eye = torch.eye(4, dtype=torch.complex128)
    delta = eye[:, 0] - rhs
    preparation = eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    circuit = fq.Circuit(4, dtype=torch.complex128).any(0, 1, matrix=preparation).h(3)
    matrix = torch.tensor(MATRIX, dtype=torch.complex128)
    for index, angle in enumerate(phases):
        if index:
            circuit.any(2, 0, 1, matrix=matrix.mH if index % 2 else matrix)
        # Native wire order: state MSB, state LSB, block, signal auxiliary.
        diagonal = [
            cmath.exp(0.5j * angle * (1 if (word & 1) ^ (not (word & 2)) else -1))
            for word in range(16)
        ]
        circuit.any(
            *range(4), matrix=torch.diag(torch.tensor(diagonal, dtype=torch.complex128))
        )
    return circuit.h(3)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().reshape(16, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
