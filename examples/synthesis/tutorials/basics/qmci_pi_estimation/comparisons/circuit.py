"""Independent one-bit integer circle predicate and dense amplification."""

import math

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit


def preparation_matrix() -> torch.Tensor:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    permutation = torch.zeros((8, 8), dtype=torch.complex128)
    for word in range(8):
        x, y = word & 1, (word >> 1) & 1
        permutation[word ^ (4 if x * x + y * y < 4 else 0), word] = 1
    return permutation @ torch.kron(
        torch.eye(2, dtype=torch.complex128), torch.kron(h, h)
    )


def reference_circuit(k: int = 1) -> fq.Circuit:
    return amplified_circuit(preparation_matrix(), k)


def run_example() -> None:
    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(k)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        print(k, program.report["resources"])
    print("Full states and absolute phases agree for zero, one and two iterations.")


if __name__ == "__main__":
    run_example()
