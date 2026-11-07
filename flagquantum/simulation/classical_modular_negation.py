# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModNeg and CModNeg.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py

"""Modular value negation on the original method's strict interior domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def modular_negate_classical_value(
    value: IntegerValue, modulus: int | Expr
) -> IntegerValue | Expr:
    """Return modulus-value only for 0 < value < modulus, otherwise preserve it."""
    if 0 < value < modulus:
        return cast("IntegerValue | Expr", modulus - value)
    return value


def controlled_modular_negate_classical_values(
    control: IntegerValue,
    value: IntegerValue,
    modulus: int | Expr,
    *,
    control_value: int = 1,
) -> tuple[IntegerValue, IntegerValue | Expr]:
    """Short-circuit inactive controls before comparisons, retaining operand identity."""
    if control == control_value:
        return control, modular_negate_classical_value(value, modulus)
    return control, value
