"""Independent direct RY and entangler Circuit, preserving absolute phase."""

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states

BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.13 + 0.11 * i) for i in range(12)))


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for layer in range(3):
        for index, wire in enumerate((3, 2, 1, 0)):
            circuit.x(wire).ry(wire, theta=params[4 * layer + index])
        circuit.cx(3, 2).cx(2, 1).cx(1, 0)
    return circuit


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("state",))


if __name__ == "__main__":
    run_example()
