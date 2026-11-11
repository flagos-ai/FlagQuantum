# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _Int/_UInt/_MontgomeryUInt scalar and array range validation.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Integer-word value validation for classical register bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sympy import Expr


def integer_word_domain(width: int | Expr, *, signed: bool = False) -> range:
    """Enumerate the source stored integer range without normalizing its width."""
    if signed:
        maximum = 1 << (width - 1)
        return range(-maximum, maximum)
    return range(cast(int, 2**width))


def validate_integer_word_value(
    value: object, width: int | Expr, *, signed: bool = False
) -> None:
    """Check a scalar word without casting or replacing its object.

    NumPy integer scalars follow Python integer bounds; arrays are not scalar
    register values. Unsigned and stored Montgomery words share the full
    width-bounded domain. Source bool values count as Python integers.
    """
    if not isinstance(value, (int, np.integer)):
        raise ValueError("Scalar integer register requires a Python or NumPy integer")
    low, high = _word_bounds(width, signed)
    if value < low or value >= high:
        raise ValueError("Classical integer value is outside the register domain")


def validate_integer_word_array(
    values: NDArray[Any], width: int | Expr, *, signed: bool = False
) -> None:
    """Retain the source range-only array checks without dtype normalization.

    Unlike scalar checks, original arrays have no integer-type guard. Floating
    arrays and NaN can pass this diagnostic validation; it does not certify that
    each value is an executable quantum word. Subclasses and empty axes remain.
    """
    low, high = _word_bounds(width, signed)
    if np.any(values < low):
        raise ValueError("Classical register array contains values below its domain")
    if np.any(values >= high):
        raise ValueError("Classical register array contains values above its domain")


def _word_bounds(
    width: int | Expr,
    signed: bool,
) -> tuple[int | float | Expr, int | float | Expr]:
    if type(signed) is not bool:
        raise ValueError("Integer word signedness must be boolean")
    if signed:
        return -(2 ** (width - 1)), 2 ** (width - 1)
    return 0, 2**width
