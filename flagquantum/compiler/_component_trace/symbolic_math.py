# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 symbolics.math_funcs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py

"""Real-valued resource arithmetic with on-demand symbolic operations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from ...core.value_math import SymbolicInt as SymbolicInt
from ...core.value_math import SymbolicReal as SymbolicReal
from ...core.value_math import smax as smax
from ...core.value_math import smin as smin

if TYPE_CHECKING:
    from sympy import Expr


def ceil(value: SymbolicReal) -> SymbolicInt:
    if isinstance(value, (int, float)):
        return math.ceil(value)
    from sympy import ceiling

    return cast("Expr", ceiling(value))


def floor(value: SymbolicReal) -> SymbolicInt:
    if isinstance(value, (int, float)):
        return math.floor(value)
    from sympy import floor as symbolic_floor

    return cast("Expr", symbolic_floor(value))


def log2(value: SymbolicReal) -> SymbolicReal:
    if isinstance(value, (int, float)):
        return math.log2(value)
    from sympy.codegen.cfunctions import log2 as symbolic_log2

    return cast("Expr", symbolic_log2(value))


def bit_length(value: SymbolicReal) -> SymbolicInt:
    """Return the bit width of the nonnegative integer part."""
    if isinstance(value, (int, float)) and 0 <= value < 1:
        return 0
    return ceil(log2(floor(value) + 1))
