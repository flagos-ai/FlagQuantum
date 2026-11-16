"""Independent marked-state reflection and dense Fourier reference."""

import math

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe


def operators() -> tuple[torch.Tensor, torch.Tensor, int]:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    preparation = torch.ones((1, 1), dtype=torch.complex128)
    for _ in range(4):
        preparation = torch.kron(preparation, h)
    oracle = torch.diag(
        torch.tensor(
            [-1 if (word & 3) + (word >> 2) <= 2 else 1 for word in range(16)],
            dtype=torch.complex128,
        )
    )
    zero = torch.eye(16, dtype=torch.complex128)
    zero[0, 0] = -1
    grover = -preparation @ zero @ preparation.mH @ oracle
    return grover, preparation, 5


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
