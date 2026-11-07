# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlScaleModAdd.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Classical controlled scaled modular accumulation without reordered arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def controlled_modular_scaled_add_classical_values(
    control: IntegerValue,
    left: IntegerValue,
    right: IntegerValue,
    constant: int | Expr,
    modulus: int | Expr,
) -> tuple[IntegerValue, IntegerValue, IntegerValue | Expr]:
    """Preserve source control checks, multiply/add/modulo order and output types.

    Unlike CModAddK, active CtrlScaleModAdd imposes no residue bounds. Inactive
    zero is returned as a Python integer before any arithmetic or checks. No
    casts or intermediate modular reductions repair NumPy overflow.
    """
    if control == 0:
        return 0, left, right
    if not control == 1:
        raise AssertionError("Bad ctrl value.")
    return (
        control,
        left,
        cast("IntegerValue | Expr", (right + left * constant) % modulus),
    )
