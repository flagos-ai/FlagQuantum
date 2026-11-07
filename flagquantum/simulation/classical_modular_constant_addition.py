# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModAddK and CModAddK.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Distinct original constant-modular-add value and control evaluation policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def modular_add_constant_classical_values(
    value: IntegerValue,
    constant: int | Expr,
    modulus: int | Expr,
    *,
    control: IntegerValue | None = None,
    control_values: int | tuple[int, ...] = (),
) -> dict[str, IntegerValue | Expr]:
    """Evaluate original ModAddK before checking packed big-endian controls.

    The candidate output is calculated even on inactive controls. Only x < mod
    is checked, with no lower bound. Targets >= mod are unchanged. Preserve the
    source control tuple conversion, binary-string decoding and assertions.
    Register width and modulus construction validation remain separate.
    """
    cvs = (
        (control_values,) if isinstance(control_values, int) else tuple(control_values)
    )
    output = (
        cast("IntegerValue | Expr", (value + constant) % modulus)
        if value < modulus
        else value
    )
    if cvs:
        if control is None:
            raise AssertionError
        if control == int("".join(str(bit) for bit in cvs), 2):
            return {"ctrl": control, "x": output}
        return {"ctrl": control, "x": value}
    if control is not None:
        raise AssertionError
    return {"x": output}


def controlled_modular_add_constant_classical_values(
    control: IntegerValue,
    value: IntegerValue,
    constant: int | Expr,
    modulus: int | Expr,
) -> tuple[IntegerValue, IntegerValue | Expr]:
    """Evaluate CModAddK with control-first bypass and strict active residue bounds.

    Inactive zero controls are returned as Python zero, as upstream. Active
    non-one controls assert before domain checks or arithmetic. This behavior
    differs from ModAddK and must not be replaced with its eager policy.
    """
    if control == 0:
        return 0, value
    if not control == 1:
        raise AssertionError("Bad ctrl value.")
    if not (0 <= value < modulus):
        raise ValueError(
            f"x={value!r} is outside the valid interval for modular addition [0, {modulus}]"
        )
    return control, cast("IntegerValue | Expr", (value + constant) % modulus)
