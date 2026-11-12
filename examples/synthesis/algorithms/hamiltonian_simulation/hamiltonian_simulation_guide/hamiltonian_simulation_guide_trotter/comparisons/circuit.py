"""Independent dense exponentials in an explicit Circuit."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_trotter.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(
    time: float = 1.0, order: int = 1, repetitions: int = 10
) -> fq.Circuit:
    dtype = torch.complex128
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    i = torch.eye(2, dtype=dtype)
    z = torch.diag(torch.tensor([1, -1], dtype=dtype))
    terms = ((0.3, torch.kron(z, z)), (0.7, torch.kron(i, x)), (0.2, torch.kron(x, i)))
    sequence = terms if order == 1 else (*terms, *reversed(terms))
    circuit = fq.Circuit(2, dtype=dtype)
    identity = torch.eye(4, dtype=dtype)
    step = time / repetitions / order
    for _ in range(repetitions):
        for coefficient, pauli in sequence:
            angle = step * coefficient
            circuit.any(
                *range(2),
                matrix=math.cos(angle) * identity - 1j * math.sin(angle) * pauli,
            )
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("qba",))


if __name__ == "__main__":
    run_example()
