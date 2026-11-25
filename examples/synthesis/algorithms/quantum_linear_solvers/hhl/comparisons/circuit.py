"""Independent analytic evolution, Fourier matrices and the retained gate network."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.data import (
    CONTROL_BITS,
    RHS,
    ROTATIONS,
)


def inversion_network(
    circuit: fq.Circuit, selector: tuple[int, ...], indicator: int
) -> None:
    for theta, bit in zip(ROTATIONS, CONTROL_BITS, strict=True):
        circuit.ry(indicator, theta=theta).cx(selector[bit], indicator)


def reference_circuit(amplitudes: tuple[float, ...] = RHS) -> fq.Circuit:
    rhs = torch.tensor(amplitudes, dtype=torch.complex128)
    rhs = rhs / torch.linalg.vector_norm(rhs)
    eye = torch.eye(4, dtype=torch.complex128)
    delta = eye[:, 0] - rhs
    preparation = (
        eye
        if torch.linalg.vector_norm(delta) < 1e-14
        else eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    )
    hamiltonian = torch.tensor(
        [
            [0.13, 0, 0, 0.01],
            [0, 0.03, 0.05, 0],
            [0, 0.05, 0.03, 0],
            [0.01, 0, 0, 0.13],
        ],
        dtype=torch.complex128,
    )
    circuit = fq.Circuit(7, dtype=torch.complex128).any(0, 1, matrix=preparation)
    for wire in range(2, 6):
        circuit.h(wire)
    controlled = []
    for bit in range(4):
        evolution = torch.matrix_exp(2j * math.pi * (1 << bit) * hamiltonian)
        matrix = torch.zeros((8, 8), dtype=torch.complex128)
        matrix[:4, :4] = eye
        matrix[4:, 4:] = evolution
        controlled.append(matrix)
        circuit.any(5 - bit, 0, 1, matrix=matrix)
    indices = torch.arange(16, dtype=torch.float64)
    inverse_dft = torch.exp(-2j * torch.pi * torch.outer(indices, indices) / 16) / 4
    circuit.any(2, 3, 4, 5, matrix=inverse_dft)
    inversion_network(circuit, (5, 4, 3, 2), 6)
    circuit.any(2, 3, 4, 5, matrix=inverse_dft.mH)
    for bit in reversed(range(4)):
        circuit.any(5 - bit, 0, 1, matrix=controlled[bit].mH)
    for wire in range(2, 6):
        circuit.h(wire)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().reshape(128, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
