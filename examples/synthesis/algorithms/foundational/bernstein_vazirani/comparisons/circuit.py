"""Check the phase-oracle form against an independently cleaned kickback qubit."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.bernstein_vazirani.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def circuit_reference(secret: int) -> fq.Circuit:
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    # Reference Z gates are the effective action of the CX kickback oracle.
    for bit in range(5):
        if secret & (1 << bit):
            circuit.z(4 - bit)
    for wire in range(5):
        circuit.h(wire)
    return circuit


def kickback_reference(secret: int) -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128).x(5).h(5)
    for wire in range(5):
        circuit.h(wire)
    for bit in range(5):
        if secret & (1 << bit):
            circuit.cx(4 - bit, 5)
    circuit.h(5).x(5)
    for wire in range(5):
        circuit.h(wire)
    return circuit


def run_example() -> None:
    for secret in (13, 22):
        program = build_program(secret)
        reference = circuit_reference(secret)
        compare_states(program, reference, ("x",))
        # Independently retain and unprepare the original oracle's extra qubit.
        kickback = fq.run(kickback_reference(secret)).statevector().reshape(32, 2)
        expected = fq.run(reference).statevector().flatten()
        torch.testing.assert_close(kickback[:, 0], expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            kickback[:, 1], torch.zeros_like(expected), atol=1e-12, rtol=0
        )
        assert program.report["workspace_qubits"] == 0


if __name__ == "__main__":
    run_example()
