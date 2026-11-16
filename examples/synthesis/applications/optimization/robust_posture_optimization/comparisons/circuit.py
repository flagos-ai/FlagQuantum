"""Check two nonzero bindings against independent native Circuit rotations."""

import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.robust_posture_optimization.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize

BINDINGS = (
    {"theta": [0.17, -0.23, 0.41, -0.59, 0.67, -0.83]},
    {"theta": [-1.13, 0.79, -0.53, 0.31, -0.19, 1.07]},
)
REGISTERS = ("q",)


def circuit_reference(theta: list[float]) -> fq.Circuit:
    # Array element zero is the least significant bit of the named register.
    circuit = fq.Circuit(2, dtype=torch.complex128)
    for wire, offset in ((1, 0), (0, 3), (0, 0)):
        circuit.rz(wire, theta=theta[offset + 2])
        circuit.ry(wire, theta=theta[offset + 1])
        circuit.rx(wire, theta=theta[offset])
    return circuit


def run_example() -> None:
    for bindings in BINDINGS:
        program = synthesize(main, classical_inputs=bindings)
        compare_states(program, circuit_reference(**bindings), REGISTERS)
        assert program.report["workspace_qubits"] == 0


if __name__ == "__main__":
    run_example()
