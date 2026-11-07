# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModSub and CModSub.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py

"""Classical modular subtraction retaining source domain and in-place behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def modular_subtract_classical_values(
    left: IntegerValue, right: IntegerValue, modulus: int | Expr
) -> tuple[IntegerValue, IntegerValue]:
    """Subtract only inside the source residue domain; otherwise return inputs.

    Original -= and += operations are retained. In particular, a mutable NumPy
    right operand may be modified, unlike modular addition's new output array.
    No Python casts, extra modulo reduction or unsigned-overflow repair occurs.
    """
    if 0 <= left < modulus and 0 <= right < modulus:
        right -= left
        if right < 0:
            right += modulus
    return left, right


def controlled_modular_subtract_classical_values(
    control: IntegerValue,
    left: IntegerValue,
    right: IntegerValue,
    modulus: int | Expr,
    *,
    control_value: int = 1,
) -> tuple[IntegerValue, IntegerValue, IntegerValue]:
    """Retain original control-first short-circuiting before residue checks."""
    if control == control_value:
        left, right = modular_subtract_classical_values(left, right, modulus)
    return control, left, right
