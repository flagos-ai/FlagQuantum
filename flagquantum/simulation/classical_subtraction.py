# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Subtract and SubtractFrom.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/subtraction.py

"""Classical subtraction directions with the original conversion boundaries."""

from __future__ import annotations

from operator import rshift
from typing import TYPE_CHECKING, cast

from .classical_addition import add_integer_values
from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def subtract_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    width: int | Expr,
    *,
    signed: bool = False,
) -> tuple[IntegerValue, int | Expr]:
    """Preserve left and replace right with left-right at the target width.

    signed reflects the source left dtype, as in original Subtract. Arithmetic
    happens before Python int conversion, retaining NumPy overflow/errors.
    Symbolic literals keep original arithmetic and result types; unresolved
    expressions retain their original modulo/conversion/shift failures.
    """
    modulus = 2**width
    if not signed:
        return left, int((left - right) % modulus)
    half = cast("int | Expr", rshift(modulus, 1))
    return left, cast("int | Expr", int((left - right + half) % modulus) - half)


def subtract_from_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    width: int | Expr,
    *,
    signed: bool = False,
) -> tuple[IntegerValue, int]:
    """Preserve left and replace right with right-left, casting first upstream.

    Unlike Subtract, original SubtractFrom converts both operands and width to
    Python integers before calling shared add_ints. Do not merge these policies.
    """
    return left, cast(
        int,
        add_integer_values(int(right), -int(left), num_bits=int(width), signed=signed),
    )
