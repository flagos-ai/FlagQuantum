# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned Qualtran v0.7.0 CAdd.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_addition.py

"""Classical controlled addition retaining inactive and NumPy source semantics."""

from typing import TYPE_CHECKING

from .classical_addition import add_integer_values
from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def controlled_add_classical_values(
    control: IntegerValue,
    left: IntegerValue,
    right: IntegerValue,
    width: "int | Expr",
    *,
    control_value: int = 1,
    signed: bool = False,
) -> tuple[IntegerValue, IntegerValue, IntegerValue]:
    """Preserve control and left, updating right only on the activation value.

    Inactive branches return the original objects before checking symbolic
    width. Active branches require a Python integer width and preserve original
    NumPy arithmetic instead of the ordinary Add consumer's integer casts.
    Unequal-width signed mathematical results may differ from upstream's
    zero-extended quantum decomposition; this function does not repair it.
    """
    if control_value not in (0, 1):
        raise ValueError("Controlled addition activation must be zero or one")
    if control != control_value:
        return control, left, right
    if not isinstance(width, int):
        raise ValueError("Active classical controlled addition requires concrete width")
    return control, left, add_integer_values(left, right, num_bits=width, signed=signed)
