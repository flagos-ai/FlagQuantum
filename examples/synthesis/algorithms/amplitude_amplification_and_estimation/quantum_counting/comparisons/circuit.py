"""Independent permutation, uniform preparation and full Grover reflection."""

import math

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit


def preparation_matrix() -> torch.Tensor:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    four = torch.kron(torch.kron(h, h), torch.kron(h, h))
    permutation = torch.zeros((32, 32), dtype=torch.complex128)
    for word in range(32):
        marked = (word & 3) + ((word >> 2) & 3) <= 2
        permutation[word ^ (16 if marked else 0), word] = 1
    return permutation @ torch.kron(torch.eye(2, dtype=torch.complex128), four)


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
