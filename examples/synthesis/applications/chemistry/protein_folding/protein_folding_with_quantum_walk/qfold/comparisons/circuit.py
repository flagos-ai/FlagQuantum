"""Independent QFold controlled rotations, shift permutation and oracle."""

import math

import torch

import flagquantum as fq

from ..physics import AMPLITUDES, ANGLES


def step_matrix(a: float = ANGLES[0], b: float = ANGLES[1]) -> torch.Tensor:
    def rotation(angle: float, mask: int, expected: int) -> torch.Tensor:
        matrix = torch.eye(16, dtype=torch.complex128)
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        for word in range(16):
            if not word & 2 and word & mask == expected:
                partner = word | 2
                matrix[word, word] = matrix[partner, partner] = c
                matrix[word, partner], matrix[partner, word] = -s, s
        return matrix

    # Obtain the complete coin operator from independent one-gate matrices.
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    identity = torch.eye(2, dtype=torch.complex128)
    flip = torch.kron(torch.eye(4, dtype=torch.complex128), torch.kron(x, identity))
    hadamard = torch.kron(torch.eye(8, dtype=torch.complex128), h)
    coin_matrix = flip @ rotation(b, 5, 1) @ rotation(a, 9, 0) @ hadamard @ flip
    shift = torch.zeros((16, 16), dtype=torch.complex128)
    for word in range(16):
        toggle = (4 if word & 1 else 8) if word & 2 else 0
        shift[word ^ toggle, word] = 1
    oracle = torch.diag(
        torch.tensor(
            [-1 if word & 3 == 0 else 1 for word in range(16)], dtype=torch.complex128
        )
    )
    return oracle @ coin_matrix.mH @ shift @ coin_matrix


def reference_circuit(
    a: float = ANGLES[0],
    b: float = ANGLES[1],
    amplitudes: tuple[float, ...] = AMPLITUDES,
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
        .any(*range(4), matrix=step_matrix(a, b))
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
        fq.run(program).statevector().flatten(), complete.flatten(), atol=1e-12, rtol=0
    )
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
