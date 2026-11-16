"""Independent exact payoff evaluation and complete dense loading unitary."""

import math
from fractions import Fraction

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit
from examples.synthesis.qpe_reference import probability_preparation

from ..physics import ANGLE_TABLE, PROBABILITIES


def preparation_matrix() -> torch.Tensor:
    rotations = torch.zeros((32, 32), dtype=torch.complex128)
    for x2 in range(4):
        for x1 in range(4):
            payoff = max(Fraction(x1), Fraction(x1 + x2 - 3, 2))
            index = payoff * 2
            if index.denominator != 1 or not 0 <= index < 8:
                raise ValueError("The exact payoff does not fit the raw selector")
            angle = ANGLE_TABLE[int(index)]
            word = x1 + 4 * x2
            c, s = math.cos(angle / 2), math.sin(angle / 2)
            rotations[word, word] = rotations[word + 16, word + 16] = c
            rotations[word, word + 16], rotations[word + 16, word] = -s, s
    loader = probability_preparation(PROBABILITIES)
    return rotations @ torch.kron(
        torch.eye(2, dtype=torch.complex128), torch.kron(loader, loader)
    )


def reference_circuit(k: int = 1) -> fq.Circuit:
    return amplified_circuit(preparation_matrix(), k)


def run_example() -> None:
    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(),
            fq.run(reference_circuit(k)).statevector().flatten(),
            atol=1e-11,
            rtol=0,
        )
        print(k, program.report["resources"])
    print("Both asset distributions and complete amplification states agree.")


if __name__ == "__main__":
    run_example()
