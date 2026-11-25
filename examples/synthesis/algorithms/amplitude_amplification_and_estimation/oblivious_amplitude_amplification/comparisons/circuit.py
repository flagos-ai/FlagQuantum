"""Independent dense Pauli SELECT and block-only reflection for OAA."""

import math

import torch

import flagquantum as fq

from ..physics import AMPLITUDES, PAULI_TERMS


def encoding_matrix() -> torch.Tensor:
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    paulis = {"X": x, "Z": z}
    select = torch.zeros((16, 16), dtype=torch.complex128)
    for word, term in enumerate(PAULI_TERMS):
        projector = torch.zeros((4, 4), dtype=torch.complex128)
        projector[word, word] = 1
        select += torch.kron(torch.kron(paulis[term[1]], paulis[term[0]]), projector)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    prepare = torch.kron(torch.eye(4, dtype=torch.complex128), torch.kron(h, h))
    return prepare @ select @ prepare


def amplification_matrix(repetitions: int = 1) -> torch.Tensor:
    encode = encoding_matrix()
    reflection = torch.diag(torch.tensor([-1, 1, 1, 1] * 4, dtype=torch.complex128))
    grover = -encode @ reflection @ encode.mH @ reflection
    power: torch.Tensor = torch.linalg.matrix_power(grover, repetitions)
    return power @ encode


def reference_circuit(
    repetitions: int = 1, amplitudes: tuple[float, ...] = AMPLITUDES
) -> fq.Circuit:
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    vector /= torch.linalg.vector_norm(vector)
    identity = torch.eye(4, dtype=torch.complex128)
    difference = identity[:, 0] - vector
    norm = difference.vdot(difference)
    prepare = (
        identity
        if float(norm.real) < 1e-30
        else identity - 2 * torch.outer(difference, difference.conj()) / norm
    )
    return (
        fq.Circuit(4, dtype=torch.complex128)
        .any(0, 1, matrix=prepare)
        .any(*range(4), matrix=amplification_matrix(repetitions))
    )


def run_example() -> None:
    from ..main import build_program

    for repetitions in (0, 1, 2):
        program = build_program(repetitions)
        torch.testing.assert_close(
            fq.run(program).statevector(),
            fq.run(reference_circuit(repetitions)).statevector(),
            atol=1e-12,
            rtol=0,
        )
        print(repetitions, program.report["resources"])
    print("Full oblivious amplification states agree without phase alignment.")


if __name__ == "__main__":
    run_example()
