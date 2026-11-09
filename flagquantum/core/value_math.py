# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 symbolics.math_funcs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py

"""Ordered neutral width arithmetic shared by metadata and resource models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, cast

from .value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr

SymbolicInt: TypeAlias = "int | Expr"
SymbolicReal: TypeAlias = "int | float | Expr"


def prod(values: Iterable[SymbolicInt]) -> SymbolicInt:
    """Multiply in iteration order, starting with the original integer identity."""
    result: SymbolicInt = 1
    for value in values:
        result = result * value
    return result


def ssum(values: Iterable[SymbolicInt]) -> SymbolicInt:
    """Add in iteration order, without coercion or a numerical reduction."""
    result: SymbolicInt = 0
    for value in values:
        result = result + value
    return result


def _extreme(
    values: tuple[SymbolicReal | Iterable[SymbolicReal], ...], *, maximum: bool
) -> SymbolicReal:
    name = "smax" if maximum else "smin"
    if not values:
        raise ValueError(f"{name} expected at least 1 argument, got 0")
    items = (
        tuple(values[0])
        if len(values) == 1 and isinstance(values[0], Iterable)
        else values
    )
    if not items:
        raise ValueError(f"{name}() arg is an empty sequence")
    scalars = cast(tuple[SymbolicReal, ...], items)
    if len(scalars) == 1:
        return scalars[0]
    # Exact built-ins have no object protocol. Subclasses must still be checked.
    concrete = all(type(value) in (int, float, bool) for value in scalars)
    if not concrete and is_symbolic_value(*scalars):
        from sympy import Max, Min

        return cast("Expr", (Max if maximum else Min)(*scalars))
    return max(scalars) if maximum else min(scalars)


def smax(*values: SymbolicReal | Iterable[SymbolicReal]) -> SymbolicReal:
    """Return the source maximum, preserving singleton identity and errors."""
    return _extreme(values, maximum=True)


def smin(*values: SymbolicReal | Iterable[SymbolicReal]) -> SymbolicReal:
    """Return the source minimum, preserving singleton identity and errors."""
    return _extreme(values, maximum=False)
