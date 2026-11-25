"""Lower static orbital mathematics to ordinary one- and two-qubit gates."""

from __future__ import annotations

import torch

from ...circuit import Circuit
from ...core.ir import CircuitIR
from .decomposition import decompose_orbital_rotation
from .fourier import fourier_decomposition
from .numerics import static_tensor
from .stabilizer import StabilizerReduction


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


def stabilizer_clifford(
    reduction: StabilizerReduction, *, inverse: bool = False
) -> Circuit:
    """Construct the reduction's native Clifford C, or C dagger for lifting.

    This circuit transforms the original register to the diagonal frame. It
    does not discard pivot qubits or implement an arbitrary reduced ansatz.
    A zero-qubit full register has no Circuit; lift_stabilizer_state handles
    its one-element vector directly. No statevector is allocated here.
    """
    if not isinstance(reduction, StabilizerReduction):
        raise TypeError("reduction must be a StabilizerReduction")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")
    if reduction.mapping.n_qubits == 0:
        raise ValueError("a zero-qubit reduction has no Clifford Circuit")
    circuit = Circuit.from_ir(
        CircuitIR(
            n_wires=reduction.mapping.n_qubits,
            dtype="complex128",
            instructions=reduction.clifford,
        )
    )
    return circuit.adjoint() if inverse else circuit


def lift_stabilizer_state(
    reduction: StabilizerReduction, state: torch.Tensor
) -> torch.Tensor:
    """Embed a reduced vector into the original selected Pauli symmetry sector.

    Apply E = C dagger J, where J is reduction.mapping's diagonal-frame affine
    embedding. The vector may be unnormalized; its norm and absolute phase are
    retained. This explicit inspection path allocates O(2**full_width) memory,
    is limited to 20 full qubits before allocation, and uses static finite CPU
    complex128 data. Autograd inputs are refused rather than detached.

    Args:
        reduction: Symbolic result returned by reduce_stabilizers.
        state: Flat vector with 2**reduction.n_qubits entries, including one
            entry when all qubits have been removed.

    Returns:
        A flat original-register vector in the selected sector.

    Raises:
        TypeError: Inputs have the wrong native type or a nonnumeric dtype.
        ValueError: Shape, finiteness, autograd or the dense budget is invalid.
    """
    if not isinstance(reduction, StabilizerReduction):
        raise TypeError("reduction must be a StabilizerReduction")
    if reduction.mapping.n_qubits > 20:
        raise ValueError("dense state embedding is limited to 20 qubits")
    vector = static_tensor(state, "reduced state", ndim=1)
    full = reduction.mapping.lift_state(vector)
    if reduction.mapping.n_qubits == 0:
        return full
    circuit = Circuit(reduction.mapping.n_qubits, dtype=torch.complex128, inputs=full)
    circuit.compose(stabilizer_clifford(reduction, inverse=True))
    return circuit.state().reshape(-1)
