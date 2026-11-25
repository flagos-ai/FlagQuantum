"""Independent dense 32-word reference and a closed-form source block."""

import cmath
import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (
    ALPHA,
    DTYPE,
    WEIGHTS,
)

from .phases import DEGREE, Factorization


def _ry(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor(((cosine, -sine), (sine, cosine)), dtype=DTYPE)


def preparation_matrix(values: tuple[float, ...]) -> torch.Tensor:
    """Construct the two real rotations controlled by the high bit."""
    a, b, c, d = values
    first = _ry(2 * math.atan2(math.hypot(c, d), math.hypot(a, b)))
    rotated = torch.zeros((4, 4), dtype=DTYPE)
    rotated[:2, :2] = _ry(2 * math.atan2(b, a))
    rotated[2:, 2:] = _ry(2 * math.atan2(d, c))
    return rotated @ torch.kron(first, torch.eye(2, dtype=DTYPE))


def dense_walk() -> torch.Tensor:
    """Construct the four-term Pauli SELECT and reflect the prepared flags."""
    x = torch.tensor(((0, 1), (1, 0)), dtype=DTYPE)
    z = torch.diag(torch.tensor((1, -1), dtype=DTYPE))
    id2 = torch.eye(2, dtype=DTYPE)
    id4 = torch.eye(4, dtype=DTYPE)
    table = (id4, torch.kron(z, id2), torch.kron(x, x), torch.kron(z, z))
    select = torch.zeros((16, 16), dtype=DTYPE)
    for key, action in enumerate(table):
        section = slice(4 * key, 4 * key + 4)
        select[section, section] = action
    preparation = torch.kron(
        preparation_matrix(tuple(math.sqrt(weight / ALPHA) for weight in WEIGHTS)), id4
    )
    reflect = torch.diag(torch.tensor((1,) * 4 + (-1,) * 12, dtype=DTYPE))
    return reflect @ preparation.mH @ select @ preparation


def reference_circuit(data: tuple[float, ...], phases: Factorization) -> fq.Circuit:
    """Use complete controlled matrix blocks on both signal branches."""
    walk = dense_walk()
    identity = torch.eye(16, dtype=DTYPE)
    first = torch.diag(torch.tensor((1, 0), dtype=DTYPE))
    second = torch.diag(torch.tensor((0, 1), dtype=DTYPE))
    forward = torch.kron(walk.contiguous(), first) + torch.kron(identity, second)
    backward = torch.kron(identity, first) + torch.kron(walk.mH.contiguous(), second)
    circuit = fq.Circuit(5, dtype=DTYPE).any(2, 3, matrix=preparation_matrix(data))
    for index, (theta, phi, lam) in enumerate(phases.rows):
        if index:
            circuit.any(0, 1, 2, 3, 4, matrix=backward if index <= DEGREE else forward)
        phases_out = torch.diag(torch.tensor((cmath.exp(1j * phi), 1), dtype=DTYPE))
        phases_in = torch.diag(torch.tensor((cmath.exp(1j * lam), 1), dtype=DTYPE))
        sign = torch.diag(torch.tensor((1, -1), dtype=DTYPE))
        circuit.any(4, matrix=phases_out @ _ry(2 * theta) @ sign @ phases_in)
    return circuit
