"""Finite complex Pauli algebra independent of measurement and execution."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Complex

Factors = tuple[tuple[int, str], ...]
_PRODUCT = {
    ("x", "y"): ("z", 1j),
    ("y", "x"): ("z", -1j),
    ("y", "z"): ("x", 1j),
    ("z", "y"): ("x", -1j),
    ("z", "x"): ("y", 1j),
    ("x", "z"): ("y", -1j),
}


def count(value: int, name: str, *, minimum: int = 0) -> int:
    """Validate an integer before coercion, including NumPy index scalars."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    result = operator.index(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def scalar(value: object) -> complex:
    """Validate a finite numeric coefficient without detaching tensors."""
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError("coefficient must be a finite numeric scalar")
    result = complex(value)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError("coefficient must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PauliTerm:
    """A complex coefficient and sorted, distinct non-identity factors."""

    coefficient: complex
    factors: Factors = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", scalar(self.coefficient))
        factors: list[tuple[int, str]] = []
        seen: set[int] = set()
        for wire, axis in self.factors:
            wire = count(wire, "qubit")
            if not isinstance(axis, str) or axis.lower() not in ("i", "x", "y", "z"):
                raise ValueError("Pauli axis must be I, X, Y or Z")
            if wire in seen:
                raise ValueError("a Pauli term cannot repeat a qubit")
            seen.add(wire)
            if axis.lower() != "i":
                factors.append((wire, axis.lower()))
        object.__setattr__(self, "factors", tuple(sorted(factors)))


@dataclass(frozen=True, slots=True)
class PauliOperator:
    """Canonical complex sum, including non-Hermitian and zero operators.

    Width is supplied separately by an encoding or execution problem, so a
    zero or identity value does not silently discard idle qubits.
    """

    terms: tuple[PauliTerm, ...] = ()

    def __post_init__(self) -> None:
        merged: dict[Factors, complex] = {}
        for term in self.terms:
            if not isinstance(term, PauliTerm):
                raise TypeError("PauliOperator requires PauliTerm values")
            merged[term.factors] = merged.get(term.factors, 0j) + term.coefficient
        object.__setattr__(
            self,
            "terms",
            tuple(
                PauliTerm(value, factors)
                for factors, value in sorted(merged.items())
                if value != 0
            ),
        )

    def __add__(self, other: object) -> PauliOperator:
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return PauliOperator(self.terms + other.terms)

    def __neg__(self) -> PauliOperator:
        return PauliOperator(
            tuple(PauliTerm(-t.coefficient, t.factors) for t in self.terms)
        )

    def __sub__(self, other: object) -> PauliOperator:
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other: object) -> PauliOperator:
        if isinstance(other, Complex) and not isinstance(other, bool):
            value = scalar(other)
            return PauliOperator(
                tuple(PauliTerm(value * t.coefficient, t.factors) for t in self.terms)
            )
        if not isinstance(other, PauliOperator):
            return NotImplemented
        terms: list[PauliTerm] = []
        for left in self.terms:
            for right in other.terms:
                factors = dict(left.factors)
                phase = left.coefficient * right.coefficient
                for wire, axis in right.factors:
                    previous = factors.get(wire)
                    if previous is None:
                        factors[wire] = axis
                    elif previous == axis:
                        del factors[wire]
                    else:
                        factors[wire], factor = _PRODUCT[previous, axis]
                        phase *= factor
                terms.append(PauliTerm(phase, tuple(factors.items())))
        return PauliOperator(tuple(terms))

    def __rmul__(self, other: object) -> PauliOperator:
        return self * other

    def dagger(self) -> PauliOperator:
        """Return the adjoint without changing the factor convention."""
        return PauliOperator(
            tuple(PauliTerm(t.coefficient.conjugate(), t.factors) for t in self.terms)
        )

    def is_hermitian(self, *, atol: float = 0.0) -> bool:
        """Check collected coefficients against an explicit absolute tolerance."""
        tolerance = scalar(atol)
        if tolerance.imag or tolerance.real < 0:
            raise ValueError("atol must be finite and nonnegative")
        return all(abs(t.coefficient.imag) <= tolerance.real for t in self.terms)


def pauli(coefficient: complex = 1.0, factors: Factors = ()) -> PauliOperator:
    """Construct a single word; ``pauli()`` is the identity."""
    return PauliOperator((PauliTerm(coefficient, factors),))
