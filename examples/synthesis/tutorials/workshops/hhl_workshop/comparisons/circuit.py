"""Independent Householder preparation, dense Fourier and table-block Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.hhl_reference import matrix_hhl_circuit
from examples.synthesis.tutorials.workshops.hhl_workshop.main import (
    build_program,
)
from examples.synthesis.tutorials.workshops.hhl_workshop.physics import (
    ANGLES,
    RAW_INPUT,
    UNITARY,
)

PHASE = (2, 3, 4, 5)
DATA = (0, 1)
INDICATOR = 6
ENVIRONMENT_BITS = 0


def reference_circuit(
    amplitudes: tuple[float, ...] = RAW_INPUT, *, workspace: int = 0
) -> fq.Circuit:
    circuit = matrix_hhl_circuit(
        7 + workspace, DATA, PHASE, INDICATOR, UNITARY, ANGLES, amplitudes
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
