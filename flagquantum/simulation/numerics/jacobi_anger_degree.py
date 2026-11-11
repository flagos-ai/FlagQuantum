# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/polynomial/jacobi_anger_approximations.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/polynomial/jacobi_anger_approximations.py

"""Source Bessel threshold search and symbolic asymptotic degree expression."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from typing import cast

import numpy as np

from ...core.value_math import SymbolicInt, SymbolicReal
from ...core.value_parameters import is_symbolic_value


def degree_jacobi_anger_approximation(
    t: SymbolicReal,
    *,
    precision: SymbolicReal,
    bessel_jn: Callable[[int, float], float] | None = None,
) -> SymbolicInt:
    """Retain the source doubling/bisect/isclose degree criterion.

    Symbolic parameters return C_JA * (t + log(1/precision)/log(log(1/precision))),
    with positive unspecified C_JA. This is an asymptotic expression, not a
    physical gate count or a numerical error certificate.

    Concrete parameters require an explicit scalar Bessel evaluator. Preserve
    source evaluation order, NumPy isclose defaults, and final assertion,
    including source assertion failures at coarse precision. Finite, valid numeric inputs are the
    supported inspection domain; the source search has no nonfinite guard or
    iteration bound. No fallback or degree correction is applied.
    """
    if is_symbolic_value(t, precision):
        import sympy

        constant = sympy.Symbol("C_{JA}", positive=True)
        return constant * (
            t + sympy.log(1 / precision) / sympy.log(sympy.log(1 / precision))
        )
    if bessel_jn is None:
        raise ValueError("concrete degree search requires a Bessel evaluator")

    def term_too_small(order: int) -> bool:
        return bool(
            np.isclose(bessel_jn(order, cast(float, t)), 0, atol=float(precision))
        )

    degree = 1
    while not term_too_small(degree):
        degree *= 2
    degree = bisect.bisect_left(range(degree), True, key=term_too_small) - 1
    assert not term_too_small(degree) and term_too_small(degree + 1)
    return degree
