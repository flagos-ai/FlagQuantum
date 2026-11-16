"""Independent complete prefix loader, threshold permutation and reflection."""

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit
from examples.synthesis.qpe_reference import probability_preparation

from ..physics import PROBABILITIES


def preparation_matrix() -> torch.Tensor:
    permutation = torch.zeros((256, 256), dtype=torch.complex128)
    for word in range(256):
        permutation[word ^ (128 if (word & 127) < 32 else 0), word] = 1
    return permutation @ torch.kron(
        torch.eye(2, dtype=torch.complex128), probability_preparation(PROBABILITIES)
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
