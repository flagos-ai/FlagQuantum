"""Independent matrix powers and dense inverse Fourier, retaining all data."""

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import amplitude_preparation, dense_qpe
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.main import (
    build_program,
)
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.physics import (
    INITIAL_VECTOR,
    UNITARY,
)

PHASE_BITS = 8
DATA_BITS = 3


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
