"""Independent complete loader, reflection and matrix-power QPE."""

import math

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe, probability_preparation

from ..main import PROBABILITIES


def operators() -> tuple[torch.Tensor, torch.Tensor, int]:
    rotations = torch.zeros((16, 16), dtype=torch.complex128)
    for word in range(8):
        angle = 0.25 * word + 0.2
        c, s = math.cos(angle), math.sin(angle)
        rotations[2 * word : 2 * word + 2, 2 * word : 2 * word + 2] = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.complex128
        )
    loader = rotations @ torch.kron(
        probability_preparation(PROBABILITIES), torch.eye(2, dtype=torch.complex128)
    )
    reflection = torch.eye(16, dtype=torch.complex128)
    reflection[0, 0] = -1
    oracle = torch.diag(torch.tensor([1, -1] * 8, dtype=torch.complex128))
    return -loader @ reflection @ loader.mH @ oracle, loader, 3


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
