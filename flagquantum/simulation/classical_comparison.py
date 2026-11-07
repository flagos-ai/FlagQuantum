# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanConstant, LessThanEqual and Equals.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Original numerical comparison predicates XORed into caller-owned target values."""

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def less_than_constant_classical_values(
    value: IntegerValue, target: IntegerValue, constant: "int | Expr"
) -> tuple[IntegerValue, IntegerValue]:
    """Preserve value and apply target XOR (value < constant), without width checks."""
    return value, cast(IntegerValue, target ^ (value < constant))


def less_equal_classical_values(
    left: IntegerValue, right: IntegerValue, target: IntegerValue
) -> tuple[IntegerValue, IntegerValue, IntegerValue]:
    """Preserve operands and apply the raw comparison/XOR operations in source order."""
    return left, right, cast(IntegerValue, target ^ (left <= right))


def equal_classical_values(
    left: IntegerValue, right: IntegerValue, target: IntegerValue
) -> tuple[IntegerValue, IntegerValue, IntegerValue]:
    """Apply the source equality predicate without dtype, range or target validation."""
    return left, right, cast(IntegerValue, target ^ (left == right))
