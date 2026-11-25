"""Independent full-state and occupation-sign validation for the source workflow."""

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
    build_program,
)

from ..matrix import occupation_matrix, reference_circuit
from ..optimization import hamiltonian, optimize


def run_example() -> None:
    """Check absolute phase and source fermionic energy in the documented mode order."""
    optimization = optimize()
    for parameters in (
        (0.0,) * 12,
        tuple(0.05 * (i + 1) for i in range(12)),
        tuple((-1) ** i * (0.13 + 0.11 * i) for i in range(12)),
        optimization.parameters,
    ):
        program = build_program(parameters)
        actual = fq.run(program).statevector().flatten()
        reference = fq.run(reference_circuit(parameters)).statevector().flatten()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
        expected = float((reference.conj() @ (occupation_matrix() @ reference)).real)
        result = fq.run(
            program.circuit, outputs=[fq.expectation(hamiltonian(), name="energy")]
        )
        assert abs(float(result.expectation("energy").reshape(())) - expected) < 1e-12
        assert program.report["workspace_qubits"] == 0
        print("Maximum full-state error:", float((actual - reference).abs().max()))
        print("Independent occupation energy:", expected)
    print("Absolute phase, mode order, fermionic signs and zero workspace agree.")


if __name__ == "__main__":
    run_example()
