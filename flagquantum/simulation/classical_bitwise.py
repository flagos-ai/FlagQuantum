# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Xor, XorK and BitwiseNot.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Classical bitwise register values, separate from compiler gate emission."""

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sympy import Expr

IntegerValue: TypeAlias = int | np.integer[Any] | NDArray[np.integer[Any]]


def xor_classical_values(
    source: IntegerValue, target: IntegerValue
) -> tuple[IntegerValue, IntegerValue]:
    """Preserve source and return its bitwise XOR into target.

    NumPy broadcasting, integer promotion and failures follow the original
    integer XOR operator. Caller-owned arrays are never modified in place.
    """
    return source, source ^ target


def xor_constant_classical_value(
    target: IntegerValue, constant: "int | Expr"
) -> IntegerValue:
    """Evaluate XorK, retaining its NumPy scalar constant conversion.

    Symbolic constants are not executable classical values, including Sympy
    integer literals. For a NumPy scalar target the original casts the constant
    to that scalar's dtype before XOR; any conversion error remains visible.
    """
    if not isinstance(constant, int):
        from sympy import Expr

        if isinstance(constant, Expr):
            raise ValueError("Cannot classically evaluate a symbolic XOR constant")
    if isinstance(target, np.integer):
        loaded = cast("np.integer[Any]", np.array(constant, dtype=target.dtype)[()])
        return target ^ loaded
    return target ^ cast(int, constant)


def bitwise_not_classical_value(
    target: IntegerValue, width: int, *, unsigned: bool
) -> IntegerValue:
    """Return -target-1, wrapping only unsigned or Montgomery register values.

    Signed and opaque register models retain negative integer results. This
    function models concrete widths; symbolic width estimates belong to the
    compiler. It preserves original NumPy promotion, broadcasting and overflow
    behavior rather than normalizing intermediate values to Python integers.
    """
    if type(width) is not int or width < 1 or type(unsigned) is not bool:
        raise ValueError("Classical NOT requires positive width and boolean encoding")
    result = -target - 1
    if unsigned:
        result %= 2**width
    return result
