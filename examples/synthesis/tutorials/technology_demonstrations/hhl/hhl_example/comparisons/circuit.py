"""Independent Householder preparation, dense Fourier and table-block Circuit."""

import math

import torch

import flagquantum as fq
from examples.synthesis.hhl_reference import matrix_hhl_circuit
from examples.synthesis.tutorials.technology_demonstrations.hhl.hhl_example.main import (
    build_program,
)
from examples.synthesis.tutorials.technology_demonstrations.hhl.hhl_example.physics import (
    ANGLES,
    RAW_INPUT,
    UNITARY,
)

PHASE = (3, 4)
DATA = (0, 1, 2)
INDICATOR = 5
ENVIRONMENT_BITS = 0


def reference_circuit(
    amplitudes: tuple[float, ...] = RAW_INPUT, *, workspace: int = 0
) -> fq.Circuit:
    norm = math.hypot(*amplitudes)
    amplitudes = tuple(value / norm for value in amplitudes)
    circuit = matrix_hhl_circuit(
        6 + workspace, DATA, PHASE, INDICATOR, UNITARY, ANGLES, amplitudes
    )
    return circuit


def run_example() -> None:
    program = build_program()
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(reference_circuit()).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    print("Complete HHL state, absolute phase and retained environment agree.")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
