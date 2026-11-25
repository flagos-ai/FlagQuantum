"""Independent guarded block rotations, complete loader and amplification."""

import math

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit
from examples.synthesis.qpe_reference import probability_preparation

from ..physics import ANGLE_TABLE, PROBABILITIES


def preparation_matrix(strike: int = 13) -> torch.Tensor:
    rotations = torch.zeros((64, 64), dtype=torch.complex128)
    for word in range(32):
        angle = ANGLE_TABLE[word] if word >= strike else 0.0
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        rotations[word, word] = rotations[word + 32, word + 32] = c
        rotations[word, word + 32], rotations[word + 32, word] = -s, s
    return rotations @ torch.kron(
        torch.eye(2, dtype=torch.complex128), probability_preparation(PROBABILITIES)
    )


def reference_circuit(k: int = 1, strike: int = 13) -> fq.Circuit:
    return amplified_circuit(preparation_matrix(strike), k)


def run_example() -> None:
    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(k)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        print(k, program.report["resources"])
    print("Original option loading and full amplification states agree.")


if __name__ == "__main__":
    run_example()
