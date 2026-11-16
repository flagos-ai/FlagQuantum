"""Independent Householder preparation, dense Fourier and table-block Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.hybrid_hhl_for_portfolio_optimization.portfolio_optimization_with_hhl.main import (
    build_program,
)
from examples.synthesis.applications.finance.hybrid_hhl_for_portfolio_optimization.portfolio_optimization_with_hhl.physics import (
    ANGLES,
    COMPARISON_INPUT,
    RAW_INPUT,
    UNITARY,
)
from examples.synthesis.hhl_reference import householder, matrix_hhl_circuit

PHASE = (2, 3, 4, 5)
DATA = (6, 7)
INDICATOR = 0
ENVIRONMENT_BITS = 8


def reference_circuit(
    amplitudes: tuple[float, ...] = RAW_INPUT,
    comparison: tuple[float, ...] = COMPARISON_INPUT,
    *,
    workspace: int = 0,
) -> fq.Circuit:
    circuit = matrix_hhl_circuit(
        10 + workspace, DATA, PHASE, INDICATOR, UNITARY, ANGLES, amplitudes
    )
    circuit.any(8, 9, matrix=householder(comparison))
    circuit.h(1).gate("cswap", (1, 6, 8)).gate("cswap", (1, 7, 9)).h(1)
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
