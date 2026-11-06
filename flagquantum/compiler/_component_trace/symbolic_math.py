# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 symbolics.math_funcs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py

"""Real-valued resource arithmetic with on-demand symbolic operations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:
    from sympy import Expr

SymbolicInt: TypeAlias = "int | Expr"
SymbolicReal: TypeAlias = "int | float | Expr"


def ceil(value: SymbolicReal) -> SymbolicInt:
    if isinstance(value, (int, float)):
        return math.ceil(value)
    from sympy import ceiling

    return cast("Expr", ceiling(value))


def floor(value: SymbolicReal) -> SymbolicInt:
    if isinstance(value, (int, float)):
        return math.floor(value)
    from sympy import floor as symbolic_floor

    return cast("Expr", symbolic_floor(value))


def log2(value: SymbolicReal) -> SymbolicReal:
    if isinstance(value, (int, float)):
        return math.log2(value)
    from sympy.codegen.cfunctions import log2 as symbolic_log2

    return cast("Expr", symbolic_log2(value))


def bit_length(value: SymbolicReal) -> SymbolicInt:
    """Return the bit width of the nonnegative integer part."""
    if isinstance(value, (int, float)) and 0 <= value < 1:
        return 0
    return ceil(log2(floor(value) + 1))


def _extreme(
    values: tuple[SymbolicReal | Iterable[SymbolicReal], ...], *, maximum: bool
) -> SymbolicReal:
    items = (
        tuple(values[0])
        if len(values) == 1 and isinstance(values[0], Iterable)
        else values
    )
    if not items:
        raise ValueError("Symbolic extrema require at least one value")
    scalars = cast("tuple[SymbolicReal, ...]", items)
    if len(scalars) == 1:
        return scalars[0]
    if all(isinstance(value, (int, float)) for value in scalars):
        numbers = cast(tuple[int | float, ...], scalars)
        return max(numbers) if maximum else min(numbers)
    from sympy import Max, Min

    return cast("Expr", (Max if maximum else Min)(*scalars))


def smax(*values: SymbolicReal | Iterable[SymbolicReal]) -> SymbolicReal:
    """Return the concrete or symbolic maximum of a pack or one iterable."""
    return _extreme(values, maximum=True)


def smin(*values: SymbolicReal | Iterable[SymbolicReal]) -> SymbolicReal:
    """Return the concrete or symbolic minimum of a pack or one iterable."""
    return _extreme(values, maximum=False)
