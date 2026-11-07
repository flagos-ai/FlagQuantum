# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModAdd and CModAdd.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Classical modular addition with original equality-boundary diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def modular_add_classical_values(
    left: IntegerValue, right: IntegerValue, modulus: int | Expr
) -> tuple[IntegerValue, IntegerValue | Expr]:
    """Preserve left and return (left+right) mod modulus under source checks.

    The original exactly-one-equals-modulus exemption and its short-circuit
    comparisons are retained, including their diagnostic behavior outside the
    clean quantum domain. No casts, normalization or array reductions are added.
    """
    special_case = (left == modulus) ^ (right == modulus)
    if not (0 <= left < modulus or special_case):
        raise ValueError(
            f"x={left!r} is outside the valid interval for modular addition [0, {modulus}]"
        )
    if not (0 <= right < modulus or special_case):
        raise ValueError(
            f"y={right!r} is outside the valid interval for modular addition [0, {modulus}]"
        )
    return left, cast("IntegerValue | Expr", (left + right) % modulus)


def controlled_modular_add_classical_values(
    control: IntegerValue,
    left: IntegerValue,
    right: IntegerValue,
    modulus: int | Expr,
    *,
    control_value: int = 1,
) -> tuple[IntegerValue, IntegerValue, IntegerValue | Expr]:
    """Return inactive operands unchanged before any arithmetic or domain check."""
    if control != control_value:
        return control, left, right
    left, output = modular_add_classical_values(left, right, modulus)
    return control, left, output
