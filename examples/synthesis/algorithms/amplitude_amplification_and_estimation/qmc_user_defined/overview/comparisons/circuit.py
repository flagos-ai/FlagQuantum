"""Independent matrices for the three native QMCI stages."""

import math
from typing import Literal

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe, probability_preparation

REFERENCE_PROBABILITIES = tuple(i / 28 for i in range(8))


def operators(
    probabilities: tuple[float, ...] = REFERENCE_PROBABILITIES,
    slope: float = 0.5,
    offset: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotations = torch.zeros((16, 16), dtype=torch.complex128)
    for word in range(8):
        angle = (slope * word + offset) / 2
        c, s = math.cos(angle), math.sin(angle)
        rotations[2 * word : 2 * word + 2, 2 * word : 2 * word + 2] = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.complex128
        )
    loader = rotations @ torch.kron(
        probability_preparation(probabilities), torch.eye(2, dtype=torch.complex128)
    )
    zero = torch.eye(16, dtype=torch.complex128)
    zero[0, 0] = -1
    marked = torch.diag(torch.tensor([1, -1] * 8, dtype=torch.complex128))
    return -loader @ zero @ loader.mH @ marked, loader


def reference_circuit(
    stage: Literal["preparation", "grover", "estimation"],
    probabilities: tuple[float, ...] = REFERENCE_PROBABILITIES,
    slope: float = 0.5,
    offset: float = 0.4,
) -> fq.Circuit:
    grover, loader = operators(probabilities, slope, offset)
    if stage == "estimation":
        return dense_qpe(grover, loader, 3)
    return fq.Circuit(4, dtype=torch.complex128).any(
        *range(4), matrix=loader if stage == "preparation" else grover
    )


def run_example() -> None:
    from ..main import build_program

    for stage in ("preparation", "grover", "estimation"):
        program = build_program(stage)
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(stage)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        assert program.report["workspace_qubits"] == 0
        print(
            stage,
            "maximum absolute state error:",
            float((actual - expected).abs().max()),
        )
    print(
        "All three complete source-precision states agree, including absolute phase and retained data."
    )


if __name__ == "__main__":
    run_example()
