"""Quantum Fourier transform circuits.

The transform maps a computational basis state ``|j>`` to
``(1/sqrt(N)) * sum_k exp(2*pi*1j*j*k/N) |k>``. This module builds that circuit from
Hadamard, controlled-phase, and swap gates.

The transform is a subroutine: it carries no performance claim of its own, and this module
does not select a runtime.

The transform exists once. :func:`qft` builds it on a fresh register over qubits
``0..n-1``, and :func:`append_qft` places that same circuit onto the caller's qubits with
:meth:`Circuit.compose <flagquantum.circuit.Circuit.compose>`, so there is one construction
of the transform rather than one per way of asking for it. The inverse transform is the
forward one's :meth:`~flagquantum.circuit.Circuit.adjoint` rather than a second reversed
emission, which is the relation the unitary satisfies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...circuit import Circuit

__all__ = ["append_qft", "qft"]


def append_qft(
    circuit: Circuit, qubits: Sequence[int], *, inverse: bool = False
) -> None:
    """Append the quantum Fourier transform on ``qubits`` to ``circuit`` in place.

    The transform is built on its own register and placed onto ``qubits`` in the order they
    are given, so it emits exactly the instructions ``circuit.compose(qft(len(qubits),
    inverse=inverse), qubits=qubits)`` emits, and ``qubits[0]`` carries the most significant
    output. The placement follows the rules of
    :meth:`Circuit.compose <flagquantum.circuit.Circuit.compose>` rather than carrying a
    second set of them: ``qubits`` must lie inside the receiver's range, and a batched
    receiver is refused rather than having this unbatched transform broadcast into it.

    Args:
        circuit: The circuit to extend.
        qubits: The qubits carrying the transform, most significant first.
        inverse: Append the inverse transform instead of the forward one.

    Raises:
        ValueError: If ``qubits`` repeats a qubit, or if
            :meth:`Circuit.compose <flagquantum.circuit.Circuit.compose>` refuses the
            placement.
    """
    ordered = list(qubits)
    if not ordered:
        return
    if len(set(ordered)) != len(ordered):
        raise ValueError("qft qubits must be distinct")
    circuit.compose(qft(len(ordered), inverse=inverse), qubits=ordered)


def qft(n_qubits: int, *, inverse: bool = False) -> Circuit:
    """Build a standalone quantum Fourier transform circuit on ``n_qubits`` qubits.

    Args:
        n_qubits: The number of qubits; must be at least one.
        inverse: Build the inverse transform instead of the forward one.

    Returns:
        A circuit using ``n`` Hadamards, ``n(n-1)/2`` controlled-phase gates, and ``n//2``
        swaps, whose qubits are numbered ``0`` to ``n_qubits - 1``.

    Raises:
        ValueError: If ``n_qubits`` is not positive.
    """
    if n_qubits < 1:
        raise ValueError(f"qft needs at least one qubit, got {n_qubits}")
    forward = _qft_circuit(n_qubits)
    return forward.adjoint() if inverse else forward


def _qft_circuit(n_qubits: int) -> Circuit:
    """Build the forward transform on a fresh register of ``n_qubits`` qubits.

    The sequence is the Hadamard and controlled-phase ladder followed by the bit-reversal
    swaps. Qubit ``p`` takes a Hadamard and then the phases controlled by every more
    significant qubit, with the phase between two qubits one position apart being ``pi/2``.

    Args:
        n_qubits: The register width; the caller has already checked it is positive.

    Returns:
        The forward transform, its qubits numbered ``0`` to ``n_qubits - 1``.
    """
    circuit = Circuit(n_qubits)
    for position in range(n_qubits):
        circuit.gate("h", position)
        for offset in range(position + 1, n_qubits):
            circuit.gate(
                "cphase",
                (offset, position),
                theta=math.pi / 2 ** (offset - position),
            )
    for position in range(n_qubits // 2):
        circuit.gate("swap", (position, n_qubits - 1 - position))
    return circuit
