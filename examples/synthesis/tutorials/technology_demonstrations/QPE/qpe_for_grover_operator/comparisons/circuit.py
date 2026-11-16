"""Independent prime-indicator preparation and complete Grover matrix."""

import math

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe


def operators() -> tuple[torch.Tensor, torch.Tensor, int]:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    four = torch.ones((1, 1), dtype=torch.complex128)
    for _ in range(4):
        four = torch.kron(four, h)
    permutation = torch.zeros((32, 32), dtype=torch.complex128)
    for word in range(32):
        destination = word ^ (16 if word & 15 in (2, 3, 5, 7, 11, 13) else 0)
        permutation[destination, word] = 1
    identity = torch.eye(32, dtype=torch.complex128)
    preparation = permutation @ torch.kron(torch.eye(2, dtype=torch.complex128), four)
    zero = identity.clone()
    zero[0, 0] = -1
    oracle = torch.diag(torch.tensor([1] * 16 + [-1] * 16, dtype=torch.complex128))
    return -preparation @ zero @ preparation.mH @ oracle, identity, 5


def reference_circuit() -> fq.Circuit:
    return dense_qpe(*operators())


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    print("Complete states, absolute phase and retained data agree.")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
