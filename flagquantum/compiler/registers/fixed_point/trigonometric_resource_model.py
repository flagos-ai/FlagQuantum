# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ArcSin and ArcTan call-graph declarations.

"""Locked trigonometric estimates, with no executable gate emission."""

from sympy import Symbol

from ..._component_trace.symbolic_math import SymbolicInt


def source_arcsine_toffoli_count(
    width: SymbolicInt,
    fractional_width: SymbolicInt,
    *,
    iterations: SymbolicInt = 4,
    degree: SymbolicInt = 4,
) -> SymbolicInt:
    """Evaluate the original Appendix D Toffoli declaration."""
    n = width
    p = n - fractional_width
    return (
        degree * (3 * n**2 + n * (6 * p + 7) - 6 * (p - 1) * p - 2)
        + iterations * (n * (15 * n + 30 * p + 23) - 30 * p * (p - 1) - 4)
        + 9 * (n + 1) * p
        + 9 * n * (n + 1) // 2
        + 6 * n**2
        + 28 * n
        - 9 * p**2
        + 2
    )


def source_arctangent_t_count(
    target_width: SymbolicInt, calls: SymbolicInt
) -> SymbolicInt:
    """Apply the original PlusEqualProduct estimate for the symbolic multiplicity."""
    return 8 * calls * target_width**2


ARCTANGENT_PRODUCT_CALLS = Symbol("_c0")
