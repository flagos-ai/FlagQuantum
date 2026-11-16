"""Independent controlled encoding blocks and eight analytic rotation matrices."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.main import (
    AMPLITUDES,
    build_program,
)
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.physics import (
    ENCODING,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    scale: float = 1.0, amplitudes: tuple[float, ...] = AMPLITUDES
) -> fq.Circuit:
    a, b = amplitudes
    norm = math.hypot(a, b)
    prep = torch.tensor(
        ((a / norm, -b / norm), (b / norm, a / norm)), dtype=torch.complex128
    )
    controlled = torch.eye(16, dtype=torch.complex128)
    controlled[:8, :8] = torch.tensor(ENCODING, dtype=torch.complex128)
    rotations = torch.zeros((16, 16), dtype=torch.complex128)
    for index in range(8):
        half = -math.pi * scale * index / 2
        c, s = math.cos(half), math.sin(half)
        rotations[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = torch.tensor(
            ((c, -s), (s, c)), dtype=torch.complex128
        )
    sign = torch.diag(torch.tensor((1, -1), dtype=torch.complex128))
    circuit = fq.Circuit(5, dtype=torch.complex128).any(0, matrix=prep)
    circuit.any(4, matrix=sign)
    circuit.any(4, 1, 2, 3, matrix=controlled)
    circuit.any(1, 2, 3, 0, matrix=rotations)
    circuit.any(4, 1, 2, 3, matrix=controlled.conj().T)
    return circuit.any(4, matrix=sign)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "ancilla", "y"))


if __name__ == "__main__":
    run_example()
