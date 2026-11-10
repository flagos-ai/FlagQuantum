"""Independent matrix powers and dense inverse Fourier, retaining all data."""

import math

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.main import (
    build_program,
)
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.physics import (
    INITIAL_VECTOR,
    UNITARY,
)

PHASE_BITS = 8
DATA_BITS = 3


def reference_circuit(amplitudes: tuple[float, ...] = INITIAL_VECTOR) -> fq.Circuit:
    circuit = fq.Circuit(PHASE_BITS + DATA_BITS, dtype=torch.complex128)
    identity = torch.eye(1 << DATA_BITS, dtype=torch.complex128)
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    delta = identity[:, 0] - vector
    norm = float(delta.abs().square().sum())
    preparation = (
        identity
        if norm < 1e-28
        else identity - 2 * torch.outer(delta, delta.conj()) / norm
    )
    data = tuple(range(PHASE_BITS, PHASE_BITS + DATA_BITS))
    circuit.any(*data, matrix=preparation)
    matrix = torch.tensor(UNITARY, dtype=torch.complex128)
    for bit in range(PHASE_BITS):
        power = torch.linalg.matrix_power(matrix, 1 << bit)
        controlled = torch.zeros(
            (2 << DATA_BITS, 2 << DATA_BITS), dtype=torch.complex128
        )
        controlled[: 1 << DATA_BITS, : 1 << DATA_BITS] = identity
        controlled[1 << DATA_BITS :, 1 << DATA_BITS :] = power
        circuit.h(PHASE_BITS - 1 - bit).any(
            PHASE_BITS - 1 - bit, *data, matrix=controlled
        )
    dimension = 1 << PHASE_BITS
    axis = torch.arange(dimension, dtype=torch.float64)
    inverse = torch.exp(
        -2j * math.pi * torch.outer(axis, axis) / dimension
    ) / math.sqrt(dimension)
    return circuit.any(*range(PHASE_BITS), matrix=inverse)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector().flatten(),
        fq.run(reference_circuit()).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    print("Matrix powers, phase labels and retained environment agree.")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
