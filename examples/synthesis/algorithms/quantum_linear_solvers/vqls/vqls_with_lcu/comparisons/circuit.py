"""Independent direct RY and entangler Circuit, preserving absolute phase."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.vqls.vqls_with_lcu.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states

BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.13 + 0.11 * i) for i in range(9)))


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    for layer in range(3):
        for index, wire in enumerate((2, 1, 0)):
            circuit.ry(wire, theta=params[3 * layer + index])
        if layer == 0:
            for control, target in ((2, 1), (2, 0), (1, 0)):
                circuit.cz(control, target)
        elif layer == 1:
            for control, target in ((0, 2), (0, 1), (1, 2)):
                circuit.cz(control, target)
    return circuit


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params), reference_circuit(params), ("system_qubits",)
        )


if __name__ == "__main__":
    run_example()
