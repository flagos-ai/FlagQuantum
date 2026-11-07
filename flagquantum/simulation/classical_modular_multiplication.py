# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModDbl and CModMulK.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Modular doubling and controlled constant-product source value methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def modular_double_classical_value(
    value: IntegerValue, modulus: int | Expr
) -> IntegerValue | Expr:
    """Retain source x+x arithmetic when x < mod, otherwise preserve the object."""
    if value < modulus:
        return cast("IntegerValue | Expr", (value + value) % modulus)
    return value


def controlled_modular_multiply_constant_classical_values(
    control: IntegerValue,
    value: IntegerValue,
    constant: int | Expr,
    modulus: int | Expr,
) -> tuple[IntegerValue, IntegerValue | Expr]:
    """Use source control truthiness then x < mod, without bit or lower-bound checks."""
    if control and value < modulus:
        return control, cast("IntegerValue | Expr", (value * constant) % modulus)
    return control, value
