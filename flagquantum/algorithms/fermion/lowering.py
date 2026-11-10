"""Lower static orbital mathematics to ordinary one- and two-qubit gates."""

from __future__ import annotations

import torch

from ...circuit import Circuit
from .decomposition import decompose_orbital_rotation


def append_two_mode(circuit: Circuit, left: int, matrix: torch.Tensor) -> None:
    """Lift adjacent orbital U to JW occupation order, including det(U)."""
    gate = torch.zeros((4, 4), dtype=torch.complex128, device=matrix.device)
    gate[0, 0] = 1
    gate[1:3, 1:3] = matrix.flip((0, 1))
    gate[3, 3] = torch.linalg.det(matrix)
    circuit.any(left, left + 1, matrix=gate, name="fermionic_orbital_rotation")


def append_orbital_rotation(circuit: Circuit, unitary: torch.Tensor) -> None:
    """Append a JW orbital rotation using static adjacent Givens factors."""
    if unitary.shape != (circuit.n_qubits, circuit.n_qubits):
        raise ValueError("orbital unitary must match the circuit width")
    decomposition = decompose_orbital_rotation(unitary)
    for qubit, phase in enumerate(decomposition.phases):
        circuit.any(
            qubit,
            matrix=torch.diag(torch.stack((torch.ones_like(phase), phase))),
            name="orbital_phase",
        )
    for step in decomposition.rotations:
        append_two_mode(circuit, step.left, step.matrix)
