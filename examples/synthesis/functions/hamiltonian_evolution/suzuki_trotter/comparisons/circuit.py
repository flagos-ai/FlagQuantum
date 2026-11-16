"""Independent dense exponentials in an explicit Circuit."""

import math

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.hamiltonian_evolution.suzuki_trotter.main import (
    build_program,
)


def reference_circuit(
    time: float = 0.37, order: int = 1, repetitions: int = 1
) -> fq.Circuit:
    dtype = torch.complex128
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    z = torch.diag(torch.tensor([1, -1], dtype=dtype))
    terms = (
        (1.5, torch.kron(torch.kron(z, x), x)),
        (0.5, torch.kron(torch.kron(z, x), y)),
    )
    sequence = terms if order == 1 else (*terms, *reversed(terms))
    circuit = fq.Circuit(3, dtype=dtype)
    identity = torch.eye(8, dtype=dtype)
    step = time / repetitions / order
    for _ in range(repetitions):
        for coefficient, pauli in sequence:
            angle = step * coefficient
            circuit.any(
                *range(3),
                matrix=math.cos(angle) * identity - 1j * math.sin(angle) * pauli,
            )
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("qba",))


if __name__ == "__main__":
    run_example()
