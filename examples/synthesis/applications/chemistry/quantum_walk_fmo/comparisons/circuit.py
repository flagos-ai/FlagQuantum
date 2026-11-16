"""Derive independent dense Pauli products from the seven-site physical matrix."""

import math

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.quantum_walk_fmo.main import (
    build_program,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.physics import (
    PAULI_ORDER,
    SITE_HAMILTONIAN,
)
from examples.synthesis.compare_states import compare_states


def product_formula(time_point: float) -> torch.Tensor:
    """Compute the declared dense product from trace projections of the matrix."""
    dtype = torch.complex128
    axes = {
        "I": torch.eye(2, dtype=dtype),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.diag(torch.tensor([1, -1], dtype=dtype)),
    }
    hamiltonian = torch.zeros((8, 8), dtype=dtype)
    hamiltonian[:7, :7] = torch.tensor(SITE_HAMILTONIAN, dtype=dtype)
    identity = torch.eye(8, dtype=dtype)
    step = identity.clone()
    reconstructed = torch.zeros_like(hamiltonian)
    for word in PAULI_ORDER:
        a, b, c = (axes[axis] for axis in word)
        pauli = torch.kron(torch.kron(a, b), c)
        coefficient = torch.trace(pauli @ hamiltonian) / 8
        assert abs(float(coefficient.imag)) < 1e-12
        reconstructed += coefficient * pauli
        angle = time_point / 10 * float(coefficient.real)
        step = (math.cos(angle) * identity - 1j * math.sin(angle) * pauli) @ step
    torch.testing.assert_close(reconstructed, hamiltonian, atol=1e-12, rtol=0)
    result: torch.Tensor = torch.linalg.matrix_power(step, 10)
    return result


def reference_circuit(time_point: float = 0.037) -> fq.Circuit:
    return fq.Circuit(3, dtype=torch.complex128).any(
        0, 1, 2, matrix=product_formula(time_point)
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("qn",))


if __name__ == "__main__":
    run_example()
