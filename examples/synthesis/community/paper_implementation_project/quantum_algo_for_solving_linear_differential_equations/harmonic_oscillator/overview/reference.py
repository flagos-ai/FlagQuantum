"""Compute independent dense Circuit and signed Taylor references."""

import math

import torch

import flagquantum as fq

from .model import GENERATOR, INITIAL, ORDER


def direct_taylor(time: float) -> torch.Tensor:
    """Apply the finite Taylor polynomial by iterating matrix-vector products."""
    generator = torch.tensor(GENERATOR, dtype=torch.float64)
    term = torch.tensor(INITIAL, dtype=torch.float64)
    solution = term.clone()
    for degree in range(1, ORDER + 1):
        term = (time / degree) * (generator @ term)
        solution += term
    return solution


def reference_circuit(time: float) -> fq.Circuit:
    """Evaluate all eight powers directly and build the full five-bit reference."""
    root_weights = torch.tensor(
        [
            math.sqrt(time**degree / math.factorial(degree))
            for degree in range(ORDER + 1)
        ],
        dtype=torch.complex128,
    )
    root_weights /= root_weights.norm()
    basis_zero = torch.zeros_like(root_weights)
    basis_zero[0] = 1
    difference = basis_zero - root_weights
    householder = torch.eye(8, dtype=torch.complex128)
    if time:
        projector = difference[:, None] * difference.conj()[None, :]
        householder -= 2 * projector / torch.vdot(difference, difference)
    controlled = torch.eye(16, dtype=torch.complex128)
    controlled[:8, :8] = householder
    generator = torch.tensor(GENERATOR, dtype=torch.complex128)
    power_select = torch.zeros((16, 16), dtype=torch.complex128)
    for degree in range(8):
        block = slice(2 * degree, 2 * degree + 2)
        power_select[block, block] = torch.linalg.matrix_power(generator, degree)
    circuit = fq.Circuit(5, dtype=torch.complex128).h(0).z(4)
    circuit.any(4, 1, 2, 3, matrix=controlled)
    circuit.any(1, 2, 3, 0, matrix=power_select)
    return circuit.any(4, 1, 2, 3, matrix=controlled).z(4)
