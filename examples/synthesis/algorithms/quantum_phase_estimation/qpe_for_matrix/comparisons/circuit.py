"""Independent matrix powers and dense inverse Fourier, retaining all data."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_for_matrix.main import (
    build_program,
)
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_for_matrix.physics import (
    INITIAL_VECTOR,
    UNITARY,
)
from examples.synthesis.qpe_reference import amplitude_preparation, dense_qpe

PHASE_BITS = 7
DATA_BITS = 2


def reference_circuit(amplitudes: tuple[float, ...] = INITIAL_VECTOR) -> fq.Circuit:
    matrix = torch.tensor(UNITARY, dtype=torch.complex128)
    return dense_qpe(matrix, amplitude_preparation(amplitudes), PHASE_BITS)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector().flatten(),
        fq.run(reference_circuit()).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    print("Matrix powers, phase labels and retained environment agree.")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
