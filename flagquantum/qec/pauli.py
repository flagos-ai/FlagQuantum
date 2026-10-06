"""Phase-free Pauli operators over arbitrary qubit indices.

The symplectic convention is the standard one: a qubit listed in ``x_qubits``
carries an ``X`` factor, a qubit listed in ``z_qubits`` carries a ``Z`` factor,
and a qubit listed in both carries ``Y``. Global phase is not tracked, so an
operator is described only up to a factor of ``+/-1`` or ``+/-i``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

_LABELS = ("i", "x", "y", "z")


def _wire_tuple(wires: Iterable[int], *, owner: str) -> tuple[int, ...]:
    """Validate one strictly increasing, duplicate-free qubit collection."""

    values = tuple(wires)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError(f"{owner} must contain integer qubit indices")
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{owner} must contain non-negative qubit indices")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{owner} must be strictly increasing without duplicates")
    return normalized


@dataclass(frozen=True, order=True)
class Pauli:
    """A phase-free Pauli operator over arbitrary qubit indices.

    A qubit listed in both ``x_qubits`` and ``z_qubits`` carries ``Y``. Ordering is
    lexicographic on ``(x_qubits, z_qubits)``, so operators sort deterministically.
    """

    x_qubits: tuple[int, ...] = ()
    z_qubits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_qubits", _wire_tuple(self.x_qubits, owner="Pauli X qubits")
        )
        object.__setattr__(
            self, "z_qubits", _wire_tuple(self.z_qubits, owner="Pauli Z qubits")
        )

    @classmethod
    def from_label(cls, label: str, qubit: int) -> Pauli:
        """Build the single-qubit operator named by ``label`` (``I``, ``X``, ``Y``, ``Z``)."""

        normalized = label.lower()
        if normalized not in _LABELS:
            raise ValueError("Pauli label must be one of I, X, Y, or Z")
        if qubit < 0:
            raise ValueError("Pauli qubit must be non-negative")
        if normalized == "i":
            return cls()
        if normalized == "x":
            return cls(x_qubits=(qubit,))
        if normalized == "z":
            return cls(z_qubits=(qubit,))
        return cls(x_qubits=(qubit,), z_qubits=(qubit,))

    @classmethod
    def from_text(cls, text: str) -> Pauli:
        """Parse the form produced by :meth:`to_text`."""

        stripped = text.strip()
        if not stripped:
            raise ValueError("Pauli text must not be empty")
        if stripped == "I":
            return cls()
        x_qubits: list[int] = []
        z_qubits: list[int] = []
        for term in stripped.split("*"):
            if not term:
                raise ValueError("Pauli text must not contain an empty term")
            label = term[0].lower()
            if label not in _LABELS or label == "i":
                raise ValueError("Pauli text terms must start with X, Y, or Z")
            digits = term[1:]
            if not digits.isdigit():
                raise ValueError("Pauli text terms must end with a qubit index")
            qubit = int(digits)
            if label in ("x", "y"):
                x_qubits.append(qubit)
            if label in ("z", "y"):
                z_qubits.append(qubit)
        return cls(x_qubits=tuple(x_qubits), z_qubits=tuple(z_qubits))

    @property
    def support(self) -> tuple[int, ...]:
        """Qubit indices carrying a non-identity factor, in ascending order."""

        return tuple(sorted(set(self.x_qubits) | set(self.z_qubits)))

    @property
    def weight(self) -> int:
        """Number of qubits carrying a non-identity factor."""

        return len(self.support)

    @property
    def is_identity(self) -> bool:
        """Whether the operator acts as the identity on every qubit."""

        return not self.x_qubits and not self.z_qubits

    def commutes_with(self, other: Pauli) -> bool:
        """Whether the two operators commute, ignoring global phase."""

        x_qubits = set(self.x_qubits)
        z_qubits = set(self.z_qubits)
        overlap = sum(1 for qubit in other.x_qubits if qubit in z_qubits)
        overlap += sum(1 for qubit in other.z_qubits if qubit in x_qubits)
        return overlap % 2 == 0

    def __mul__(self, other: Pauli) -> Pauli:
        """Compose two operators, dropping the phase of the product."""

        return Pauli(
            x_qubits=tuple(sorted(set(self.x_qubits) ^ set(other.x_qubits))),
            z_qubits=tuple(sorted(set(self.z_qubits) ^ set(other.z_qubits))),
        )

    def to_text(self) -> str:
        """Render a deterministic text form such as ``X0*Z2`` or ``I``."""

        x_qubits = set(self.x_qubits)
        z_qubits = set(self.z_qubits)
        terms = []
        for qubit in self.support:
            if qubit in x_qubits and qubit in z_qubits:
                terms.append(f"Y{qubit}")
            elif qubit in x_qubits:
                terms.append(f"X{qubit}")
            else:
                terms.append(f"Z{qubit}")
        return "*".join(terms) if terms else "I"


__all__ = ("Pauli",)
