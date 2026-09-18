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

    wire: int


@dataclass(frozen=True)
class _ControlledPhase:
    """One controlled phase in a transform sequence."""

    control: int
    target: int
    angle: float


@dataclass(frozen=True)
class _Swap:
    """One wire exchange in a transform sequence."""

    left: int
    right: int


_Step = _Hadamard | _ControlledPhase | _Swap


def append_qft(
    circuit: Circuit, wires: Sequence[int], *, inverse: bool = False
) -> None:
    """Append the quantum Fourier transform on ``wires`` to ``circuit`` in place.

    The sequence is the Hadamard and controlled-phase ladder followed by the bit-reversal
    swaps. With ``inverse`` set, the same gates are emitted in reverse order and every
    controlled-phase angle is negated.

    Args:
        circuit: The circuit to extend.
        wires: The wires carrying the transform, most significant first.
        inverse: Append the inverse transform instead of the forward one.

    Raises:
        ValueError: If ``wires`` contains a repeated wire.
    """
    ordered = list(wires)
    n_wires = len(ordered)
    if n_wires == 0:
        return
    if len(set(ordered)) != n_wires:
        raise ValueError("qft wires must be distinct")

    sequence: list[_Step] = []
    for position in range(n_wires):
        sequence.append(_Hadamard(ordered[position]))
        for offset in range(position + 1, n_wires):
            sequence.append(
                _ControlledPhase(
                    control=ordered[offset],
                    target=ordered[position],
                    angle=math.pi / 2 ** (offset - position),
                )
            )
    for position in range(n_wires // 2):
        sequence.append(_Swap(ordered[position], ordered[n_wires - 1 - position]))

    if inverse:
        sequence.reverse()

    for step in sequence:
        if isinstance(step, _Hadamard):
            circuit.gate("h", step.wire)
        elif isinstance(step, _ControlledPhase):
            angle = -step.angle if inverse else step.angle
            circuit.gate("cphase", (step.control, step.target), theta=angle)
        else:
            circuit.gate("swap", (step.left, step.right))


def qft(n_wires: int, *, inverse: bool = False) -> Circuit:
    """Build a standalone quantum Fourier transform circuit on ``n_wires`` wires.

    Args:
        n_wires: The number of wires; must be at least one.
        inverse: Build the inverse transform instead of the forward one.

    Returns:
        A circuit using ``n`` Hadamards, ``n(n-1)/2`` controlled-phase gates, and ``n//2``
        swaps.

    Raises:
        ValueError: If ``n_wires`` is not positive.
    """
    if n_wires < 1:
        raise ValueError(f"qft needs at least one wire, got {n_wires}")
    circuit = Circuit(n_wires)
    append_qft(circuit, list(range(n_wires)), inverse=inverse)
    return circuit
