# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PlusEqualProduct.apply and on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py

"""Classical multiply-accumulate with the source arithmetic conversion order."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def accumulate_product_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    result: IntegerValue,
    result_width: int | Expr,
    *,
    adjoint: bool = False,
) -> tuple[IntegerValue, IntegerValue, IntegerValue | Expr]:
    """Preserve operands and add/subtract their product modulo the result width.

    Multiplication, direction multiplication, addition and modulo occur in that
    order, without Python integer casts. NumPy promotion, overflow and symbolic
    expressions therefore retain the original behavior. Width and operand-domain
    validation belong to register binding, not this numerical evaluation.
    """
    output = (result + left * right * ((-1) ** adjoint)) % (2**result_width)
    return left, right, cast("IntegerValue | Expr", output)
