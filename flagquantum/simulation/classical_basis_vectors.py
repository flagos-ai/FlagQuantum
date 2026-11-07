# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ZVector and _IntVector value methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Classical computational-basis creation and checked consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def bit_basis_classical_values(
    bit: object,
    *,
    value: object = None,
    state: bool = True,
    width: int = 1,
) -> dict[str, int]:
    """Retain the source single-bit width guard, truthiness and assertions."""
    if width != 1:
        raise NotImplementedError("Only the original single-bit vector is supported")
    bit_int = 1 if bit else 0
    if state:
        assert value is None
        return {"q": bit_int}
    assert value == bit_int, value
    return {}


def _validate_integer_vector(value: IntegerValue | Expr, width: int | Expr) -> None:
    # Source checks Expr specifically, not the broader symbolic-object protocol.
    try:
        from sympy import Expr
    except ModuleNotFoundError as error:
        if error.name != "sympy":
            raise
    else:
        if isinstance(value, Expr) or isinstance(width, Expr):
            return
    if value < 0:
        raise ValueError("Integer basis value must be nonnegative")
    if value >= 2**width:
        raise ValueError("Integer basis value exceeds its width")


def integer_basis_classical_values(
    value: IntegerValue | Expr,
    width: int | Expr,
    *,
    incoming: IntegerValue | Expr | None = None,
    state: bool = True,
) -> dict[str, IntegerValue | Expr]:
    """Preserve constructor validation before state/effect callback assertions.

    A state returns the original configured value; an effect consumes only a
    matching incoming value. No word encoding, bit emission or quantum
    projection occurs here. Symbolic Expr metadata retains the source bypass.
    """
    _validate_integer_vector(value, width)
    if state:
        assert incoming is None
        return {"val": value}
    assert incoming == value, incoming
    return {}
