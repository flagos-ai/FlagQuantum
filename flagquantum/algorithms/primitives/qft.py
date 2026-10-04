"""Quantum Fourier transform circuits.

The transform maps a computational basis state ``|j>`` to
``(1/sqrt(N)) * sum_k exp(2*pi*1j*j*k/N) |k>``. This module builds that circuit from
Hadamard, controlled-phase, and swap gates.

The transform is a subroutine: it carries no performance claim of its own, and this module
does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ...circuit import Circuit

__all__ = ["append_qft", "qft"]


@dataclass(frozen=True)
class _Hadamard:
    """One Hadamard gate in a transform sequence."""

    qubit: int


@dataclass(frozen=True)
class _ControlledPhase:
    """One controlled phase in a transform sequence."""

    control: int
    target: int
    angle: float


@dataclass(frozen=True)
class _Swap:
    """One qubit exchange in a transform sequence."""

    left: int
    right: int


_Step = _Hadamard | _ControlledPhase | _Swap


def append_qft(
    circuit: Circuit, qubits: Sequence[int], *, inverse: bool = False
) -> None:
    """Append the quantum Fourier transform on ``qubits`` to ``circuit`` in place.

    The sequence is the Hadamard and controlled-phase ladder followed by the bit-reversal
    swaps. With ``inverse`` set, the same gates are emitted in reverse order and every
    controlled-phase angle is negated.

    Args:
        circuit: The circuit to extend.
        qubits: The qubits carrying the transform, most significant first.
        inverse: Append the inverse transform instead of the forward one.

    Raises:
        ValueError: If ``qubits`` contains a repeated qubit.
    """
    ordered = list(qubits)
    n_qubits = len(ordered)
    if n_qubits == 0:
        return
    if len(set(ordered)) != n_qubits:
        raise ValueError("qft qubits must be distinct")

    sequence: list[_Step] = []
    for position in range(n_qubits):
        sequence.append(_Hadamard(ordered[position]))
        for offset in range(position + 1, n_qubits):
            sequence.append(
                _ControlledPhase(
                    control=ordered[offset],
                    target=ordered[position],
                    angle=math.pi / 2 ** (offset - position),
                )
            )
    for position in range(n_qubits // 2):
        sequence.append(_Swap(ordered[position], ordered[n_qubits - 1 - position]))

    if inverse:
        sequence.reverse()

    for step in sequence:
        if isinstance(step, _Hadamard):
            circuit.gate("h", step.qubit)
        elif isinstance(step, _ControlledPhase):
            angle = -step.angle if inverse else step.angle
            circuit.gate("cphase", (step.control, step.target), theta=angle)
        else:
            circuit.gate("swap", (step.left, step.right))


def qft(n_qubits: int, *, inverse: bool = False) -> Circuit:
    """Build a standalone quantum Fourier transform circuit on ``n_qubits`` qubits.

    Args:
        n_qubits: The number of qubits; must be at least one.
        inverse: Build the inverse transform instead of the forward one.

    Returns:
        A circuit using ``n`` Hadamards, ``n(n-1)/2`` controlled-phase gates, and ``n//2``
        swaps.

    Raises:
        ValueError: If ``n_qubits`` is not positive.
    """
    if n_qubits < 1:
        raise ValueError(f"qft needs at least one qubit, got {n_qubits}")
    circuit = Circuit(n_qubits)
    append_qft(circuit, list(range(n_qubits)), inverse=inverse)
    return circuit
