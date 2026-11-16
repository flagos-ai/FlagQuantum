"""Independent Householder preparation, dense Fourier and table-block Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.portfolio_optimization_hhl.hhl_portfolio.main import (
    build_program,
)
from examples.synthesis.applications.finance.portfolio_optimization_hhl.hhl_portfolio.physics import (
    ANGLES,
    RAW_INPUT,
    UNITARY,
)
from examples.synthesis.hhl_reference import matrix_hhl_circuit

PHASE = (5, 6, 7, 8, 9, 10)
DATA = (1, 2, 3, 4)
INDICATOR = 0
ENVIRONMENT_BITS = 0


def reference_circuit(
    amplitudes: tuple[float, ...] = RAW_INPUT, *, workspace: int = 0
) -> fq.Circuit:
    circuit = matrix_hhl_circuit(
        11 + workspace, DATA, PHASE, INDICATOR, UNITARY, ANGLES, amplitudes
    )
    return circuit


def run_example() -> None:
    program = build_program()
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(reference_circuit(workspace=3)).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    print("Complete HHL state, absolute phase and retained environment agree.")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
