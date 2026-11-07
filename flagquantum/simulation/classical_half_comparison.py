# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _HalfComparisonBase.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Half-comparison logical values, independent of physical output port bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .classical_addition import add_integer_values
from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def _comparison(
    left: IntegerValue, right: IntegerValue, relation: str
) -> bool | np.bool_ | NDArray[np.bool_]:
    if relation == ">":
        return left > right
    if relation == "<":
        return left < right
    if relation == ">=":
        return left >= right
    return left <= right


def half_comparison_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    width: int | Expr,
    *,
    relation: Literal[">", "<", ">=", "<="] = ">",
    uncompute: bool = False,
    difference: IntegerValue | None = None,
    target: IntegerValue | None = None,
) -> dict[str, IntegerValue | Expr]:
    """Compute raw difference/flag or erase the correctly supplied logical outputs.

    Source explicitly casts operands but not width before shared add_ints. The
    difference and target are logical outputs, not clean work. Erasure verifies
    both values and returns original inputs; quantum adjoint port aliases remain
    the compiler's separate authority.
    """
    if relation in (">", "<="):
        value = add_integer_values(
            -int(left), int(right), num_bits=cast(int, width + 1), signed=False
        )
    else:
        value = add_integer_values(
            int(left), -int(right), num_bits=cast(int, width + 1), signed=False
        )
    if uncompute:
        assert difference == value
        assert target == _comparison(left, right, relation)
        return {"a": left, "b": right}
    assert difference is None
    assert target is None
    return {
        "a": left,
        "b": right,
        "c": value,
        "target": int(_comparison(left, right, relation)),
    }
