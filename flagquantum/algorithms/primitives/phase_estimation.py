"""Phase estimation over a controlled unitary.

The counting register is prepared in a uniform superposition, the unitary is applied in
controlled powers whose exponents are the counting bits, and the inverse Fourier transform on
the counting register turns the accumulated phase into a readable integer.

The construction is phase estimation, attributed to A. Yu. Kitaev, "Quantum measurements and
the Abelian Stabilizer Problem", arXiv:quant-ph/9511026 (1995), a preprint; the circuit form
built here is the one recorded by Brassard, Høyer, Mosca, and Tapp, "Quantum Amplitude
Amplification and Estimation", *AMS Contemporary Mathematics* **305**, 53–74 (2002),
DOI 10.1090/conm/305/05215, arXiv:quant-ph/0005055.

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


@dataclass(frozen=True, kw_only=True)
class PhaseEstimationSpec:
    """The resolution a phase-estimation circuit achieves.

    Args:
        n_counting_qubits: The number of qubits in the counting register.
        n_evaluation_qubits: The number of qubits the operator acts on.
    """

    n_counting_qubits: int
    n_evaluation_qubits: int

    def __post_init__(self) -> None:
        """Reject a register width that cannot resolve a phase.

        Raises:
            ValueError: If ``n_counting_qubits`` is not positive, or ``n_evaluation_qubits``
                is negative.
        """
        if self.n_counting_qubits < 1:
            raise ValueError(
                "the counting register needs at least one qubit, "
                f"got {self.n_counting_qubits}"
            )
        if self.n_evaluation_qubits < 0:
            raise ValueError(
                "the evaluation register cannot have a negative width, "
                f"got {self.n_evaluation_qubits}"
            )

    @property
    def precision(self) -> float:
        """The phase resolution in radians, ``2*pi / 2**n_counting_qubits``."""
        # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
        # negative exponent yields a float, and an ``Any`` return fails the type gate.
        return 2 * math.pi / 2.0**self.n_counting_qubits

    @property
    def success_probability(self) -> float:
        """The baseline lower bound on reading the nearest phase, ``4/pi**2``.

        The bound is over every phase the circuit may be asked to resolve, not the value
        for any one instance.
        """
        return 4 / math.pi**2

    def phase_from_counts(self, counts: Mapping[str, int]) -> float:
        """Return the most frequent counting outcome read as a phase in ``[0, 1)``.

        The keys are the full register as a big-endian bit string, so the counting register
        is the first ``n_counting_qubits`` characters. Every key's count is accumulated onto
        those leading bits and the mode of that folded distribution is what is read: taking
        the most frequent full-register key instead would follow the evaluation register
        wherever it is correlated with the counting register, and return a silently wrong
        phase rather than an error. What the readout still assumes is the key layout only:
        one character per qubit, with the counting register's bits leading. The evaluation
        register's bits are otherwise read nowhere.

        Args:
            counts: Sample counts keyed by the full register's bit string.

        Returns:
            The most frequent value of the leading ``n_counting_qubits`` bits, divided by
            ``2**n_counting_qubits``.

        Raises:
            ValueError: If ``counts`` is empty, or if a key does not carry one bit per
                register qubit.
        """
        if not counts:
            raise ValueError("counts must not be empty")
        expected = self.n_counting_qubits + self.n_evaluation_qubits
        folded: dict[str, int] = {}
        for key, count in counts.items():
            if len(key) != expected:
                raise ValueError(
                    f"count key {key!r} has {len(key)} bits, "
                    f"expected {expected} for this register"
                )
            counting_key = key[: self.n_counting_qubits]
            folded[counting_key] = folded.get(counting_key, 0) + count
        most_frequent = max(folded, key=folded.__getitem__)
        # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
        # negative exponent yields a float, and an ``Any`` return fails the type gate.
        return int(most_frequent, 2) / 2.0**self.n_counting_qubits


def append_phase_estimation(
    circuit: Circuit,
    *,
    unitary: ControlledUnitary,
    counting_qubits: Sequence[int],
    evaluation_qubits: Sequence[int],
) -> None:
    """Append phase estimation for ``unitary`` to ``circuit`` in place.

    The counting qubits carry the uniform superposition and take the controlled powers of
    ``unitary``; the evaluation qubits carry the operator's eigenstate, which the caller
    prepares. The counting qubits must be the register's leading block, starting at qubit 0,
    so that they are both its most significant bits and the bits the readout reads first.

    Args:
        circuit: The circuit to extend.
        unitary: The operator whose eigenphase is estimated.
        counting_qubits: The qubits of the counting register, ``0`` upward in significance
            order.
        evaluation_qubits: The qubits the operator acts on.

    Raises:
        ValueError: If ``counting_qubits`` is empty, repeats a qubit, or is not the leading
            block ``0..len(counting_qubits)-1``; if ``evaluation_qubits`` does not carry
            ``unitary.n_qubits`` qubits; or if the two registers share a qubit.
    """
    counting = list(counting_qubits)
    evaluation = list(evaluation_qubits)
    if not counting:
        raise ValueError("the counting register needs at least one qubit, got 0")
    if len(set(counting)) != len(counting):
        raise ValueError("counting qubits must be distinct")
    if counting != list(range(len(counting))):
        raise ValueError(
            "counting qubits must be the leading block of the register, starting at qubit 0; "
            f"got {counting!r}"
        )
    if len(evaluation) != unitary.n_qubits:
        raise ValueError(
            f"the operator acts on {unitary.n_qubits} qubits, "
            f"got {len(evaluation)} evaluation qubits"
        )
    if set(counting) & set(evaluation):
        raise ValueError("counting and evaluation qubits must not overlap")

    for qubit in counting:
        circuit.gate("h", qubit)
    for position, qubit in enumerate(counting):
        power = 2 ** (len(counting) - 1 - position)
        unitary.apply_power_controlled(circuit, qubit, evaluation, power)
    append_qft(circuit, counting, inverse=True)


def phase_estimation_circuit(
    *, unitary: ControlledUnitary, n_counting_qubits: int
) -> Circuit:
    """Build a phase-estimation circuit for ``unitary``.

    The counting register occupies the first ``n_counting_qubits`` qubits and the operator's
    qubits follow it.

    Args:
        unitary: The operator whose eigenphase is estimated.
        n_counting_qubits: The number of qubits in the counting register; must be at least one.

    Returns:
        A circuit over ``n_counting_qubits + unitary.n_qubits`` qubits.

    Raises:
        ValueError: If ``n_counting_qubits`` is not positive.
    """
    if n_counting_qubits < 1:
        raise ValueError(
            f"phase estimation needs at least one counting qubit, got {n_counting_qubits}"
        )
    n_evaluation_qubits = unitary.n_qubits
    circuit = Circuit(n_counting_qubits + n_evaluation_qubits)
    append_phase_estimation(
        circuit,
        unitary=unitary,
        counting_qubits=list(range(n_counting_qubits)),
        evaluation_qubits=list(
            range(n_counting_qubits, n_counting_qubits + n_evaluation_qubits)
        ),
    )
    return circuit
