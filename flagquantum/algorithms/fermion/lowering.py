"""Lower static orbital mathematics to ordinary one- and two-qubit gates."""

from __future__ import annotations

import torch

from ...circuit import Circuit
from .decomposition import decompose_orbital_rotation
from .fourier import fourier_decomposition


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


def fermionic_fourier_transform(n_modes: int, *, inverse: bool = False) -> Circuit:
    """Build a normalized fermionic Fourier circuit in full JW occupation order.

    The forward one-particle matrix is exp(-2j*pi*p*q/n_modes)/sqrt(n_modes).
    Mode zero is the most significant occupation bit. Output modes retain
    natural order; the vacuum has phase one and all particle sectors preserve
    the determinant phase. Routing uses FSWAP, including its minus sign on |11>.

    Args:
        n_modes: Positive power of two, including one.
        inverse: Realize the adjoint Fourier transform when True.

    Returns:
        A static complex128 CPU circuit of adjacent two-qubit matrix gates.
        Append it to a prepared state using Circuit.compose. It is not a qubit
        QFT, and does not directly support parity, BK or tapered registers.

    Raises:
        TypeError: Width is not an integer or inverse is not a bool.
        ValueError: Width is not a positive power of two.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.fermion.lowering import fermionic_fourier_transform
        >>> circuit = fq.Circuit(2, dtype=torch.complex128)
        >>> _ = circuit.x(0)
        >>> _ = circuit.compose(fermionic_fourier_transform(2))
        >>> [round(p, 6) for p in circuit.state().abs().square().reshape(-1).tolist()]
        [0.0, 0.5, 0.5, 0.0]
    """
    decomposition = fourier_decomposition(n_modes, inverse=inverse)
    circuit = Circuit(len(decomposition.phases), dtype=torch.complex128)
    # Radix-2 plans have unit initial phases; the two-mode factors carry all
    # twiddle and determinant phases, including fermionic swap signs.
    for step in decomposition.rotations:
        append_two_mode(circuit, step.left, step.matrix)
    return circuit
