"""Symbolic observables and Pauli generators, independent of output requests."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from numbers import Real
from typing import Any, cast
from typing import get_type_hints as _class_type_hints

from flagquantum.core._qubit_aliases import OMITTED, Omitted, warn_qubit_alias


@dataclass(frozen=True, slots=True)
class _PauliTerm:
    coefficient: float
    factors: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class Observable:
    """An immutable real linear combination of Pauli products.

    Examples:
        >>> import flagquantum as fq
        >>> observable = 0.5 * (fq.X(0) @ fq.X(1)) - fq.Z(0)
        >>> len(observable.terms)
        2
    """

    terms: tuple[_PauliTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("an observable requires at least one Pauli term")

    def __matmul__(self, other: object) -> "Observable":
        if not isinstance(other, Observable):
            return NotImplemented
        products: list[_PauliTerm] = []
        for left in self.terms:
            for right in other.terms:
                left_qubits = {wire for wire, _ in left.factors}
                right_qubits = {wire for wire, _ in right.factors}
                if left_qubits & right_qubits:
                    raise ValueError("Pauli tensor products require disjoint qubits")
                factors = tuple(sorted((*left.factors, *right.factors)))
                products.append(
                    _PauliTerm(left.coefficient * right.coefficient, factors)
                )
        return Observable(tuple(products))

    def __add__(self, other: object) -> "Observable":
        if not isinstance(other, Observable):
            return NotImplemented
        return Observable((*self.terms, *other.terms))

    def __sub__(self, other: object) -> "Observable":
        if not isinstance(other, Observable):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> "Observable":
        return Observable(
            tuple(_PauliTerm(-term.coefficient, term.factors) for term in self.terms)
        )

    def __mul__(self, scalar: object) -> "Observable":
        value = _real_scalar(scalar)
        if value is None:
            return NotImplemented
        return Observable(
            tuple(
                _PauliTerm(value * term.coefficient, term.factors)
                for term in self.terms
            )
        )

    def __rmul__(self, scalar: object) -> "Observable":
        return self * scalar


def _real_scalar(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("observable coefficients must be finite")
    return result


def _qubit(owner: str, value: Any, *, noun: str = "qubit") -> int:
    """Read one qubit label without quietly rewriting or iterating the caller's value.

    A label is an integer, read through the interpreter's own ``__index__`` protocol,
    which admits NumPy integers and zero-dimensional torch integer tensors while refusing
    floats and strings.  ``bool`` is refused although it satisfies ``operator.index``, for
    the reason the package's counts give: it is a flag rather than a label.  ``TypeError``
    reports a wrong Python type, which is what the errors-module boundary reserves for it;
    a negative label is a magnitude and keeps its own error below.

    ``noun`` names the label in those messages, because an oscillator degree is read by
    this same rule and must not be reported as a qubit.
    """

    if isinstance(value, bool):
        raise TypeError(f"{owner} {noun} must be an integer, got {value!r}")
    try:
        label = operator.index(value)
    except TypeError:
        raise TypeError(f"{owner} {noun} must be an integer, got {value!r}") from None
    if label < 0:
        raise ValueError(f"{owner} {noun} must be a non-negative integer")
    return label


def _qubits(
    qubits: Iterable[int] | int, *, owner: str, noun: str = "qubit", unique: bool = True
) -> tuple[int, ...]:
    """Read a selection of qubit labels, one label at a time.

    A single label and a sequence of labels are both accepted, and the order matters: a
    label is tried first, so a bare zero-dimensional tensor is read as the label it
    denotes, and a value that cannot be iterated at all -- a float is the case that
    occurs -- is reported as the bad label it is instead of as ``'float' object is not
    iterable``.  ``str`` is excluded from the sequence form, because ``"01"`` is a
    mistyped label rather than two labels.

    ``unique=False`` admits a repeated label, which is what an oscillator degree list
    needs: ``ad[k] ad[k]`` is a power rather than two operators, so the group is a
    multiset.  A qubit selection keeps the default, where a repeat is a mistake.
    """

    normalized: tuple[int, ...]
    if isinstance(qubits, (str, bytes)):
        normalized = (_qubit(owner, qubits, noun=noun),)
    else:
        try:
            normalized = (_qubit(owner, qubits, noun=noun),)
        except TypeError as scalar_error:
            if not hasattr(qubits, "__iter__"):
                raise
            try:
                items = tuple(cast("Iterable[int]", qubits))
            except TypeError:
                # An iterable that refuses to be iterated as a sequence -- a
                # zero-dimensional tensor is the case that occurs -- is a bad label, not
                # a bad sequence, so the scalar refusal is the true one.
                raise scalar_error from None
            normalized = tuple(_qubit(owner, qubit, noun=noun) for qubit in items)
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{owner} {noun}s must be unique")
    return normalized


def _pauli(axis: str, qubit: int) -> Observable:
    return Observable((_PauliTerm(1.0, ((_qubit("observable", qubit), axis),)),))


def _selected_qubit(
    qubit: int | None | Omitted, wire: int | None | Omitted, *, owner: str
) -> int | None:
    """Resolve the canonical ``qubit`` argument against the deprecated ``wire`` alias.

    Supplying both spellings is an error rather than a precedence rule, matching the
    selection rule the count outputs already use.  ``None`` stays meaningful for the
    identity observable, so a resolved ``None`` is returned rather than treated as
    missing.
    """

    if not isinstance(wire, Omitted):
        if not isinstance(qubit, Omitted):
            raise TypeError(f"pass {owner} qubit or deprecated wire, not both")
        warn_qubit_alias("wire", "qubit", stacklevel=4)
        return wire
    return None if isinstance(qubit, Omitted) else qubit


def _required_qubit(
    qubit: int | None | Omitted, wire: int | None | Omitted, *, owner: str
) -> int:
    selected = _selected_qubit(qubit, wire, owner=owner)
    if selected is None:
        raise TypeError(f"{owner} requires a qubit")
    return selected


def I(
    qubit: int | None | Omitted = OMITTED, *, wire: int | None | Omitted = OMITTED
) -> Observable:
    """Return the identity observable.

    ``qubit`` may be supplied for readable Hamiltonian notation but does not affect the
    mathematical value.
    """

    selected = _selected_qubit(qubit, wire, owner="observable")
    if selected is not None:
        _qubit("observable", selected)
    return Observable((_PauliTerm(1.0, ()),))


def X(
    qubit: int | None | Omitted = OMITTED, *, wire: int | None | Omitted = OMITTED
) -> Observable:
    """Return the Pauli-X observable on ``qubit``."""

    return _pauli("x", _required_qubit(qubit, wire, owner="observable"))


def Y(
    qubit: int | None | Omitted = OMITTED, *, wire: int | None | Omitted = OMITTED
) -> Observable:
    """Return the Pauli-Y observable on ``qubit``."""

    return _pauli("y", _required_qubit(qubit, wire, owner="observable"))


def Z(
    qubit: int | None | Omitted = OMITTED, *, wire: int | None | Omitted = OMITTED
) -> Observable:
    """Return the Pauli-Z observable on ``qubit``."""

    return _pauli("z", _required_qubit(qubit, wire, owner="observable"))


_FERMION_EXPORTS = (
    "FermionOperator",
    "FermionTerm",
    "annihilate",
    "create",
    "jordan_wigner",
    "number",
    "parity_encoding",
)


_SUBMODULES = ("boson",)


__all__ = ("I", "Observable", "X", "Y", "Z", *_SUBMODULES, *_FERMION_EXPORTS)


def __getattr__(name: str) -> Any:
    # The fermionic algebra imports `Observable` and the Pauli term it builds on from this
    # module, so it can only be imported once this module has finished defining them.
    # Resolving the names on first use is what keeps that direction legal in either import
    # order, and it keeps a name an author of this module did not export an error.
    if name in _SUBMODULES:
        return import_module(f".{name}", __package__)
    if name in _FERMION_EXPORTS:
        return getattr(import_module(".fermion", __package__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Preserve historical import identities for pickles and public introspection.
# Resolve field types before preserving historical public class module names.
# Dataclass constructor/field contracts stay unchanged; introspection needs no facade import.
for _public_hint_class in (Observable,):
    _public_field_hints = _class_type_hints(_public_hint_class, include_extras=True)
    _public_hint_class.__annotations__ = {
        name: _public_field_hints[name] for name in _public_hint_class.__annotations__
    }
del _public_hint_class, _public_field_hints

for _identity_export in (_PauliTerm, Observable, I, X, Y, Z):
    _identity_export.__module__ = "flagquantum.observables"
del _identity_export
