# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Add, AddIntoPhaseGrad, SumOfSquares, and Product.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/phase_gradient.py

# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py

"""Declared arithmetic costs before selecting an IR implementation."""

from .._component_trace.resource_model import ComponentCost, GateCost


def source_add_cost(
    left_width: int, right_width: int, *, legacy_shims: bool = False
) -> ComponentCost:
    """Preserve the source Add formula, including its one-bit Clifford anomaly."""
    if (
        type(left_width) is not int
        or type(right_width) is not int
        or not 1 <= left_width <= right_width
    ):
        raise ValueError("Add widths must satisfy 1 <= left <= right")
    carries = right_width - 1
    return ComponentCost(
        "Add",
        left_width + right_width + carries,
        GateCost(
            and_bloq=carries,
            clifford=(right_width - 2) * 6 + 3 + carries * (3 if legacy_shims else 1),
            measurement=carries,
        ),
    )


def source_phase_add_cost(
    input_width: int, gradient_width: int, *, controlled: bool = False
) -> ComponentCost:
    """Use the source abstract Toffoli model for either sign and any shift."""
    if (
        type(input_width) is not int
        or input_width < 1
        or type(gradient_width) is not int
        or gradient_width < 1
        or type(controlled) is not bool
    ):
        raise ValueError(
            "Phase addition requires positive widths and a boolean control mode"
        )
    return ComponentCost(
        "AddIntoPhaseGrad",
        max(3, input_width + gradient_width + controlled),
        GateCost(toffoli=(gradient_width - 2) * (1 + controlled)),
    )


def source_sum_of_squares_cost(width: int, count: int) -> ComponentCost:
    """Count the source unsigned input words and right-only result."""
    if any(type(n) is not int or n < 1 for n in (width, count)):
        raise ValueError("Sum of squares requires a positive width and count")
    result_width = 2 * width + (count - 1).bit_length()
    return ComponentCost(
        "SumOfSquares",
        count * width + result_width,
        GateCost(toffoli=count * width**2 - width - (count % 3 == 0)),
    )


def source_product_cost(left: int, right: int) -> ComponentCost:
    """Count the source unsigned product with a twice-maximum-width result."""
    if any(type(n) is not int or n < 1 for n in (left, right)):
        raise ValueError("Product requires positive input widths")
    longest = max(left, right)
    return ComponentCost(
        "Product",
        left + right + 2 * longest,
        GateCost(toffoli=2 * left * right - longest),
    )
