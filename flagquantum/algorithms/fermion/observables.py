"""Explicit boundary from complex algebra to native measurement values."""

from __future__ import annotations

from ...core.algebra.pauli import PauliOperator
from ...observables import I, Observable, X, Y, Z


def to_observable(operator: PauliOperator, *, atol: float = 1e-10) -> Observable:
    """Convert collected Hermitian coefficients; refuse a non-Hermitian value."""
    if not isinstance(operator, PauliOperator):
        raise TypeError("operator must be a native PauliOperator")
    if not operator.is_hermitian(atol=atol):
        raise ValueError(
            "a non-Hermitian Pauli operator cannot be measured as Observable"
        )
    result = 0.0 * I()
    axes = {"x": X, "y": Y, "z": Z}
    for term in operator.terms:
        word = I()
        for wire, axis in term.factors:
            word = word @ axes[axis](wire)
        result = result + term.coefficient.real * word
    return result
