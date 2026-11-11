"""Explicit optional operator conversion; native workflows never import upstream."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from importlib import import_module
from itertools import product
from types import ModuleType
from typing import TYPE_CHECKING, SupportsIndex

from ..core.algebra.pauli import PauliOperator, PauliTerm, count, scalar
from ..observables.fermion import (
    FermionOperator,
    FermionTerm,
    annihilate,
    create,
)

if TYPE_CHECKING:
    from openfermion import FermionOperator as OpenFermionOperator
    from openfermion import QubitOperator as OpenQubitOperator


def _upstream() -> ModuleType:
    try:
        return import_module("openfermion")
    except ImportError as error:
        raise ImportError(
            "OpenFermion object conversion requires a separately installed "
            "OpenFermion package; install openfermion in your interoperability "
            "environment. Native FlagQuantum workflows do not require it."
        ) from error


def _terms(source: object, class_name: str) -> Mapping[object, object]:
    upstream = _upstream()
    if not isinstance(source, getattr(upstream, class_name)):
        raise TypeError(f"source must be an OpenFermion {class_name}")
    terms: object = getattr(source, "terms", None)
    if not isinstance(terms, Mapping):
        raise TypeError("OpenFermion terms must be a mapping")
    return terms


def _factors(term: object) -> tuple[tuple[object, object], ...]:
    if not isinstance(term, tuple):
        raise TypeError("an operator term must be a tuple of factor pairs")
    factors: list[tuple[object, object]] = []
    for factor in term:
        if not isinstance(factor, tuple) or len(factor) != 2:
            raise TypeError("an operator factor must be a pair")
        factors.append((factor[0], factor[1]))
    return tuple(factors)


def _index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an integer, not bool")
    return count(operator.index(value), name)


def from_openfermion_fermion(source: object) -> FermionOperator:
    """Convert ordered upstream ladder products to the native canonical algebra.

    Requires an installed OpenFermion package only when called. Invalid source
    types or factors raise TypeError/ValueError; coefficients must be finite.
    The input and its mutable term dictionary are never modified or retained.
    """
    terms: list[FermionTerm] = []
    for raw_term, raw_coefficient in _terms(source, "FermionOperator").items():
        coefficient = scalar(raw_coefficient)
        product = FermionOperator((FermionTerm(coefficient),))
        for raw_mode, raw_action in _factors(raw_term):
            mode = _index(raw_mode, "mode")
            action = _index(raw_action, "ladder action")
            if action not in (0, 1):
                raise ValueError("ladder action must be 0 (annihilate) or 1 (create)")
            product = product * (create(mode) if action else annihilate(mode))
        terms.extend(product.terms)
    return FermionOperator(tuple(terms))


def from_openfermion_interaction(source: object) -> FermionOperator:
    """Import number-conserving spin-orbital coefficients without pruning.

    The two-body coefficient multiplies ``a†[p] a†[q] a[r] a[s]`` directly:
    no integral permutation or additional factor of one half is applied.
    Requires OpenFermion only when called. Dense numeric NumPy arrays must
    have matching square one-body and four-index two-body shapes. Malformed
    objects, unsupported tensor keys/dtypes and nonfinite coefficients raise
    TypeError or ValueError. No external arrays are modified or retained.

    Examples:
        >>> import numpy as np
        >>> from openfermion import InteractionOperator
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> source = InteractionOperator(
        ...     0.25, np.diag([1.0, 2.0]), np.zeros((2, 2, 2, 2))
        ... )
        >>> native = from_openfermion_interaction(source)
        >>> encoded = encode_fermion(native, n_modes=2, encoding="jordan_wigner")
    """
    upstream = _upstream()
    if not isinstance(source, upstream.InteractionOperator):
        raise TypeError("source must be an OpenFermion InteractionOperator")
    tensors = getattr(source, "n_body_tensors", None)
    if not isinstance(tensors, Mapping):
        raise TypeError("InteractionOperator n_body_tensors must be a mapping")
    if set(tensors) != {(), (1, 0), (1, 1, 0, 0)}:
        raise ValueError(
            "InteractionOperator requires constant, one-body and two-body tensors only"
        )
    # NumPy is owned by the explicitly installed upstream boundary, never Core.
    numpy = import_module("numpy")
    one, two = tensors[(1, 0)], tensors[(1, 1, 0, 0)]
    if not isinstance(one, numpy.ndarray) or not isinstance(two, numpy.ndarray):
        raise TypeError("InteractionOperator tensors must be dense NumPy arrays")
    if one.ndim != 2 or one.shape[0] != one.shape[1]:
        raise ValueError("InteractionOperator one-body tensor must be square")
    width = one.shape[0]
    if two.shape != (width,) * 4:
        raise ValueError(
            "InteractionOperator two-body shape must match the one-body width"
        )
    if one.dtype.kind not in "iufc" or two.dtype.kind not in "iufc":
        raise TypeError(
            "InteractionOperator tensors require real or complex numeric dtypes"
        )
    terms = [FermionTerm(scalar(tensors[()]))]
    for p, q in product(range(width), repeat=2):
        coefficient = scalar(one[p, q])
        if coefficient:
            terms.append(FermionTerm(coefficient, (p,), (q,)))
    for p, q, r, s in product(range(width), repeat=4):
        coefficient = scalar(two[p, q, r, s])
        if coefficient:
            monomial = (
                coefficient * create(p) * create(q) * annihilate(r) * annihilate(s)
            )
            terms.extend(monomial.terms)
    return FermionOperator(tuple(terms))


def to_openfermion_fermion(source: FermionOperator) -> OpenFermionOperator:
    """Export native monomials without changing their ordered ladder products.

    Upstream may use a different normal-order convention. Compare exported
    operators physically rather than comparing their raw term dictionaries.
    Raises TypeError for a nonnative source and ImportError if upstream is absent.
    """
    if not isinstance(source, FermionOperator):
        raise TypeError("source must be a native FermionOperator")
    upstream = _upstream()
    result = upstream.FermionOperator()
    # Populate a fresh dictionary: upstream addition may prune small coefficients.
    for term in source.terms:
        factors = tuple((mode, 1) for mode in term.creations) + tuple(
            (mode, 0) for mode in term.annihilations
        )
        result.terms[factors] = term.coefficient
    return result


def from_openfermion_qubit(source: object) -> PauliOperator:
    """Convert upstream Pauli products, preserving complex and identity terms.

    Non-Hermitian operators remain algebraic values. Observable conversion is
    a separate native boundary. Malformed factors and nonfinite values refuse.
    """
    terms: list[PauliTerm] = []
    for raw_term, raw_coefficient in _terms(source, "QubitOperator").items():
        factors = []
        for raw_mode, raw_axis in _factors(raw_term):
            mode = _index(raw_mode, "qubit")
            if not isinstance(raw_axis, str) or raw_axis not in ("X", "Y", "Z"):
                raise ValueError("an upstream Pauli axis must be X, Y or Z")
            factors.append((mode, raw_axis.lower()))
        terms.append(PauliTerm(scalar(raw_coefficient), tuple(factors)))
    return PauliOperator(tuple(terms))


def to_openfermion_qubit(source: PauliOperator) -> OpenQubitOperator:
    """Export a native complex Pauli sum to an explicitly optional upstream object.

    Raises TypeError for a nonnative source and ImportError if upstream is absent.
    Neither algebra stores idle register width; consumers supply it explicitly.
    """
    if not isinstance(source, PauliOperator):
        raise TypeError("source must be a native PauliOperator")
    upstream = _upstream()
    result = upstream.QubitOperator()
    for term in source.terms:
        factors = tuple((mode, axis.upper()) for mode, axis in term.factors)
        result.terms[factors] = term.coefficient
    return result
