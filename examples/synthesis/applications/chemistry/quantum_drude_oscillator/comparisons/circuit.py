"""Use direct Circuit RY gates and explicit logical-to-physical bit order."""

from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.quantum_drude_oscillator.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(params: Sequence[float]) -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128)
    quarter = 1.570796326794897
    for layer, first in enumerate((0, 2, 4, 1, 3, 0, 2, 4, 1, 3)):
        a, b = 5 - first, 4 - first
        p0, p1, p2, p3 = params[4 * layer : 4 * layer + 4]
        circuit.rz(a, theta=quarter).rz(b, theta=quarter).ry(b, theta=quarter)
        circuit.cx(b, a)
        circuit.ry(a, theta=p0).rz(a, theta=p1)
        circuit.ry(b, theta=p2).rz(b, theta=p3)
        circuit.cx(b, a).ry(b, theta=-quarter)
        circuit.rz(a, theta=-quarter).rz(b, theta=-quarter)
    return circuit


def run_example() -> None:
    for scale in (0.37, -0.81):
        params = [scale * 0.1 * (index + 1) for index in range(40)]
        compare_states(build_program(params), reference_circuit(params), ("state",))


if __name__ == "__main__":
    run_example()
