"""Independent complete Lanchester Circuit, including all clean work bits."""

import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.hhl_lanchester.main import (
    build_program,
)
from examples.synthesis.applications.physical_systems.hhl_lanchester.physics import (
    ANGLES,
    INITIAL_VECTOR,
    UNITARY,
)
from examples.synthesis.hhl_reference import matrix_hhl_circuit


def reference_circuit(amplitudes: tuple[float, ...] = INITIAL_VECTOR) -> fq.Circuit:
    return matrix_hhl_circuit(
        16, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11), 0, UNITARY, ANGLES, amplitudes
    )


def run_example() -> None:
    program = build_program()
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(reference_circuit()).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    print("Both named fields and the complete joint HHL state agree.")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
