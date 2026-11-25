"""Independent dense Fourier SELECT and complete controller loading unitary."""

import math

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import probability_preparation

from ..physics import PROBABILITIES, ZERO_INPUT


def lcu_matrix(probabilities: tuple[float, ...] = PROBABILITIES) -> torch.Tensor:
    axis = torch.arange(4, dtype=torch.float64)
    fourier = torch.exp(2j * math.pi * torch.outer(axis, axis) / 4) / 2
    select = torch.eye(16, dtype=torch.complex128)
    select[4:8, 4:8] = fourier
    select[8:12, 8:12] = fourier.mH
    prepare = torch.kron(
        probability_preparation(probabilities), torch.eye(4, dtype=torch.complex128)
    )
    return prepare.mH @ select @ prepare


def reference_circuit(
    probabilities: tuple[float, ...] = PROBABILITIES,
    amplitudes: tuple[float, ...] = ZERO_INPUT,
) -> fq.Circuit:
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    vector /= torch.linalg.vector_norm(vector)
    identity = torch.eye(4, dtype=torch.complex128)
    difference = identity[:, 0] - vector
    norm = difference.vdot(difference)
    loading = (
        identity
        if float(norm.real) < 1e-30
        else identity - 2 * torch.outer(difference, difference.conj()) / norm
    )
    return (
        fq.Circuit(4, dtype=torch.complex128)
        .any(2, 3, matrix=loading)
        .any(*range(4), matrix=lcu_matrix(probabilities))
    )


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    expected = fq.run(reference_circuit()).statevector().flatten()
    work = program.report["workspace_qubits"]
    if not isinstance(work, int):
        raise TypeError("Expected an integer workspace estimate")
    complete = torch.zeros((16, 1 << work), dtype=torch.complex128)
    complete[:, 0] = expected
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), complete.flatten(), atol=1e-11, rtol=0
    )
    print(program.report["resources"])
    print("Full coherent LCU state agrees; predicate workspace is zero.")


if __name__ == "__main__":
    run_example()
