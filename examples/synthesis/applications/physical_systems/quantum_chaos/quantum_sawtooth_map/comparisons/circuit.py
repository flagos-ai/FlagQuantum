"""Independent signed quadratic phases and explicit dense Fourier matrices."""

import cmath
import math

import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(num_kicks: int = 2) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=torch.complex128).x(0).x(1)
    fourier = torch.tensor(
        [
            [
                cmath.exp(math.tau * 1j * row * column / 8) / math.sqrt(8)
                for column in range(8)
            ]
            for row in range(8)
        ],
        dtype=torch.complex128,
    )
    values = [raw if raw < 4 else raw - 8 for raw in range(8)]
    for _ in range(num_kicks):
        circuit.any(0, 1, 2, matrix=fourier)
        circuit.any(
            0,
            1,
            2,
            matrix=torch.diag(
                torch.tensor(
                    [cmath.exp(0.1j * math.pi * value**2 / 8) for value in values],
                    dtype=torch.complex128,
                )
            ),
        )
        circuit.any(0, 1, 2, matrix=fourier.conj().T)
        circuit.any(
            0,
            1,
            2,
            matrix=torch.diag(
                torch.tensor(
                    [cmath.exp(-1j * math.pi * value**2 / 8) for value in values],
                    dtype=torch.complex128,
                )
            ),
        )
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("p",))


if __name__ == "__main__":
    run_example()
