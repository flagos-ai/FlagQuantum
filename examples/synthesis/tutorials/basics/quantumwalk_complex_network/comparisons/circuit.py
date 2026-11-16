"""Independent conditional diffusion matrices and register swaps."""

import cmath

import torch

import flagquantum as fq


def reference_circuit(steps: int = 3, angle: float = 0.0) -> fq.Circuit:
    probabilities = torch.tensor(
        [[0, 1 / 3, 1 / 3, 1 / 3], [0.5, 0, 0.5, 0], [0.5, 0.5, 0, 0], [1, 0, 0, 0]],
        dtype=torch.float64,
    )
    vectors = probabilities.sqrt().to(torch.complex128)
    initial = vectors.flatten() / 2
    # A Householder map prepares this one initial state independently of the
    # compiler's conditional RY/CX preparation decomposition.
    axis = torch.zeros(16, dtype=torch.complex128)
    axis[0] = 1
    axis -= initial
    preparation = torch.eye(16, dtype=torch.complex128) - 2 * torch.outer(
        axis, axis.conj()
    ) / torch.vdot(axis, axis)
    circuit = fq.Circuit(4, dtype=torch.complex128).any(*range(4), matrix=preparation)
    phases = torch.diag(
        torch.tensor(
            [
                cmath.exp(1j * angle * ((x & 1) - 0.7 * (y >> 1)))
                for x in range(4)
                for y in range(4)
            ],
            dtype=torch.complex128,
        )
    )
    circuit.any(*range(4), matrix=phases)
    coin = torch.zeros((16, 16), dtype=torch.complex128)
    for index, vector in enumerate(vectors):
        coin[4 * index : 4 * index + 4, 4 * index : 4 * index + 4] = 2 * torch.outer(
            vector, vector.conj()
        ) - torch.eye(4, dtype=torch.complex128)
    for _ in range(steps):
        circuit.any(*range(4), matrix=coin)
        circuit.swap(0, 2).swap(1, 3)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    state = fq.run(program).statevector().reshape(16, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
