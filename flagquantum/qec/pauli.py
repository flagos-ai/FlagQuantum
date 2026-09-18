"""Phase-free Pauli operators over arbitrary wire indices.

The symplectic convention is the standard one: a wire listed in ``x_wires``
carries an ``X`` factor, a wire listed in ``z_wires`` carries a ``Z`` factor,
and a wire listed in both carries ``Y``. Global phase is not tracked, so an
operator is described only up to a factor of ``+/-1`` or ``+/-i``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

_LABELS = ("i", "x", "y", "z")


def _wire_tuple(wires: Iterable[int], *, owner: str) -> tuple[int, ...]:
    """Validate one strictly increasing, duplicate-free wire collection."""

    values = tuple(wires)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError(f"{owner} must contain integer wire indices")
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{owner} must contain non-negative wire indices")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{owner} must be strictly increasing without duplicates")
    return normalized


@dataclass(frozen=True, order=True)
class Pauli:
    """A phase-free Pauli operator over arbitrary wire indices.

    A wire listed in both ``x_wires`` and ``z_wires`` carries ``Y``. Ordering is
    lexicographic on ``(x_wires, z_wires)``, so operators sort deterministically.
    """

    x_wires: tuple[int, ...] = ()
    z_wires: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_wires", _wire_tuple(self.x_wires, owner="Pauli X wires")
        )
        object.__setattr__(
            self, "z_wires", _wire_tuple(self.z_wires, owner="Pauli Z wires")
        )

    @classmethod
    def from_label(cls, label: str, wire: int) -> Pauli:
        """Build the single-wire operator named by ``label`` (``I``, ``X``, ``Y``, ``Z``)."""

        normalized = label.lower()
        if normalized not in _LABELS:
            raise ValueError("Pauli label must be one of I, X, Y, or Z")
        if wire < 0:
            raise ValueError("Pauli wire must be non-negative")
        if normalized == "i":
            return cls()
        if normalized == "x":
            return cls(x_wires=(wire,))
        if normalized == "z":
            return cls(z_wires=(wire,))
        return cls(x_wires=(wire,), z_wires=(wire,))

    @classmethod
    def from_text(cls, text: str) -> Pauli:
        """Parse the form produced by :meth:`to_text`."""

        stripped = text.strip()
        if not stripped:
            raise ValueError("Pauli text must not be empty")
        if stripped == "I":
            return cls()
        x_wires: list[int] = []
        z_wires: list[int] = []
        for term in stripped.split("*"):
            if not term:
                raise ValueError("Pauli text must not contain an empty term")
            label = term[0].lower()
            if label not in _LABELS or label == "i":
                raise ValueError("Pauli text terms must start with X, Y, or Z")
            digits = term[1:]
            if not digits.isdigit():
                raise ValueError("Pauli text terms must end with a wire index")
            wire = int(digits)
            if label in ("x", "y"):
                x_wires.append(wire)
            if label in ("z", "y"):
                z_wires.append(wire)
        return cls(x_wires=tuple(x_wires), z_wires=tuple(z_wires))

    @property
    def support(self) -> tuple[int, ...]:
        """Wire indices carrying a non-identity factor, in ascending order."""

        return tuple(sorted(set(self.x_wires) | set(self.z_wires)))

    @property
    def weight(self) -> int:
        """Number of wires carrying a non-identity factor."""

        return len(self.support)

    @property
    def is_identity(self) -> bool:
        """Whether the operator acts as the identity on every wire."""

        return not self.x_wires and not self.z_wires

    def commutes_with(self, other: Pauli) -> bool:
        """Whether the two operators commute, ignoring global phase."""

        x_wires = set(self.x_wires)
        z_wires = set(self.z_wires)
        overlap = sum(1 for wire in other.x_wires if wire in z_wires)
        overlap += sum(1 for wire in other.z_wires if wire in x_wires)
        return overlap % 2 == 0

    def __mul__(self, other: Pauli) -> Pauli:
        """Compose two operators, dropping the phase of the product."""

        return Pauli(
            x_wires=tuple(sorted(set(self.x_wires) ^ set(other.x_wires))),
            z_wires=tuple(sorted(set(self.z_wires) ^ set(other.z_wires))),
        )

    def to_text(self) -> str:
        """Render a deterministic text form such as ``X0*Z2`` or ``I``."""

        x_wires = set(self.x_wires)
        z_wires = set(self.z_wires)
        terms = []
        for wire in self.support:
            if wire in x_wires and wire in z_wires:
                terms.append(f"Y{wire}")
            elif wire in x_wires:
                terms.append(f"X{wire}")
            else:
                terms.append(f"Z{wire}")
        return "*".join(terms) if terms else "I"


__all__ = ("Pauli",)
