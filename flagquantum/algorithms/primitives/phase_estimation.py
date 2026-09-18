"""Phase estimation over a controlled unitary.

The counting register is prepared in a uniform superposition, the unitary is applied in
controlled powers whose exponents are the counting bits, and the inverse Fourier transform on
the counting register turns the accumulated phase into a readable integer.

The resolution the circuit achieves is a property of the circuit, not evidence of a
performance advantage, and the cost of preparing the operator's eigenstate is not counted here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...circuit import Circuit
from .qft import append_qft
from .types import ControlledUnitary

__all__ = [
    "PhaseEstimationSpec",
    "append_phase_estimation",
    "phase_estimation_circuit",
]


@dataclass(frozen=True)
class PhaseEstimationSpec:
    """The resolution a phase-estimation circuit achieves.

    Args:
        n_counting_wires: The number of wires in the counting register.
        n_evaluation_wires: The number of wires the operator acts on.
    """

    n_counting_wires: int
    n_evaluation_wires: int

    def __post_init__(self) -> None:
        """Reject a register width that cannot resolve a phase.

        Raises:
            ValueError: If ``n_counting_wires`` is not positive, or ``n_evaluation_wires``
                is negative.
        """
        if self.n_counting_wires < 1:
            raise ValueError(
                "the counting register needs at least one wire, "
                f"got {self.n_counting_wires}"
            )
        if self.n_evaluation_wires < 0:
            raise ValueError(
                "the evaluation register cannot have a negative width, "
                f"got {self.n_evaluation_wires}"
            )

    @property
    def precision(self) -> float:
        """The phase resolution in radians, ``2*pi / 2**n_counting_wires``."""
        # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
        # negative exponent yields a float, and an ``Any`` return fails the type gate.
        return 2 * math.pi / 2.0**self.n_counting_wires

    @property
    def success_probability(self) -> float:
        """The baseline probability that the counting register reads the nearest phase."""
        return 4 / math.pi**2

    def phase_from_counts(self, counts: Mapping[str, int]) -> float:
        """Return the most frequent count read as a phase in ``[0, 1)``.

        The keys are the full register as a big-endian bit string, so the counting register
        is the first ``n_counting_wires`` characters.

        Args:
            counts: Sample counts keyed by the full register's bit string.

        Returns:
            The most frequent counting value divided by ``2**n_counting_wires``.

        Raises:
            ValueError: If ``counts`` is empty, or if the most frequent key does not carry
                one bit per register wire.
        """
        if not counts:
            raise ValueError("counts must not be empty")
        most_frequent = max(counts, key=counts.__getitem__)
        expected = self.n_counting_wires + self.n_evaluation_wires
        if len(most_frequent) != expected:
            raise ValueError(
                f"count key {most_frequent!r} has {len(most_frequent)} bits, "
                f"expected {expected} for this register"
            )
        # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
        # negative exponent yields a float, and an ``Any`` return fails the type gate.
        return (
            int(most_frequent[: self.n_counting_wires], 2) / 2.0**self.n_counting_wires
        )


def append_phase_estimation(
    circuit: Circuit,
    *,
    unitary: ControlledUnitary,
    counting_wires: Sequence[int],
    evaluation_wires: Sequence[int],
) -> None:
    """Append phase estimation for ``unitary`` to ``circuit`` in place.

    The counting wires carry the uniform superposition and take the controlled powers of
    ``unitary``; the evaluation wires carry the operator's eigenstate, which the caller
    prepares. The counting wires come first in the register, so they are its most
    significant bits.

    Args:
        circuit: The circuit to extend.
        unitary: The operator whose eigenphase is estimated.
        counting_wires: The wires of the counting register, most significant first.
        evaluation_wires: The wires the operator acts on.

    Raises:
        ValueError: If ``counting_wires`` repeats a wire or is not in increasing order,
            if ``evaluation_wires`` does not carry ``unitary.n_wires`` wires, or if the
            two registers share a wire.
    """
    counting = list(counting_wires)
    evaluation = list(evaluation_wires)
    if len(set(counting)) != len(counting):
        raise ValueError("counting wires must be distinct")
    if counting != sorted(counting):
        raise ValueError(
            "counting wires must be in increasing order of significance; "
            f"got {counting!r}"
        )
    if len(evaluation) != unitary.n_wires:
        raise ValueError(
            f"the operator acts on {unitary.n_wires} wires, "
            f"got {len(evaluation)} evaluation wires"
        )
    if set(counting) & set(evaluation):
        raise ValueError("counting and evaluation wires must not overlap")

    for wire in counting:
        circuit.gate("h", wire)
    for position, wire in enumerate(counting):
        power = 2 ** (len(counting) - 1 - position)
        unitary.apply_power_controlled(circuit, wire, evaluation, power)
    append_qft(circuit, counting, inverse=True)


def phase_estimation_circuit(
    *, unitary: ControlledUnitary, n_counting_wires: int
) -> Circuit:
    """Build a phase-estimation circuit for ``unitary``.

    The counting register occupies the first ``n_counting_wires`` wires and the operator's
    wires follow it.

    Args:
        unitary: The operator whose eigenphase is estimated.
        n_counting_wires: The number of wires in the counting register; must be at least one.

    Returns:
        A circuit over ``n_counting_wires + unitary.n_wires`` wires.

    Raises:
        ValueError: If ``n_counting_wires`` is not positive.
    """
    if n_counting_wires < 1:
        raise ValueError(
            f"phase estimation needs at least one counting wire, got {n_counting_wires}"
        )
    n_evaluation_wires = unitary.n_wires
    circuit = Circuit(n_counting_wires + n_evaluation_wires)
    append_phase_estimation(
        circuit,
        unitary=unitary,
        counting_wires=list(range(n_counting_wires)),
        evaluation_wires=list(
            range(n_counting_wires, n_counting_wires + n_evaluation_wires)
        ),
    )
    return circuit
