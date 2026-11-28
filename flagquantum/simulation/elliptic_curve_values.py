# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Value formulas from locked Qualtran v0.7.0 ecc/ec_point.py and ecc/ec_add.py.

"""Evaluate affine and Montgomery values without quantum construction guards."""

from __future__ import annotations

from typing import cast

from ..core.value_math import SymbolicInt
from .montgomery_values import (
    integer_to_montgomery_value,
    montgomery_inverse_value,
    montgomery_product_values,
    montgomery_to_integer_value,
)


def affine_point_add_values(
    left: tuple[SymbolicInt, SymbolicInt],
    right: tuple[SymbolicInt, SymbolicInt],
    modulus: SymbolicInt,
    curve_a: SymbolicInt,
) -> tuple[SymbolicInt, SymbolicInt]:
    """Retain the source infinity sentinel, inverse cast and affine formulas.

    Callers own curve consistency. Invalid inverses and symbolic integer casts
    propagate; no point validity test or coordinate normalization is added.
    """
    x, y = left
    other_x, other_y = right
    if left == (other_x, -other_y % modulus):
        return 0, 0
    if left == (0, 0):
        return right
    if right == (0, 0):
        return left
    if left == right:
        numerator = 3 * x**2 + curve_a
        denominator = 2 * y
    else:
        numerator = other_y - y
        denominator = other_x - x
    inverse = pow(int(denominator % modulus), -1, int(modulus))
    slope = ((numerator % modulus) * inverse) % modulus
    result_x = (slope**2 - x - other_x) % modulus
    result_y = (slope * (x - result_x) - y) % modulus
    return result_x, result_y


def addition_flags_values(
    modulus: int, a: int, b: int, x: int, y: int
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Return f1/f2/f3/f4/ctrl and unchanged raw words in source port order."""
    f1 = int(a == x)
    f2 = int(b == -y % modulus)
    f3 = int(a == b == 0)
    f4 = int(x == y == 0)
    ctrl = int(f2 == f3 == f4 == 0)
    return f1, f2, f3, f4, ctrl, a, b, x, y


def addition_slope_values(
    width: int,
    modulus: int,
    f1: int,
    ctrl: int,
    a: int,
    b: int,
    x: int,
    y: int,
    lam_r: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Produce the source RIGHT slope without allocating physical history."""
    x = (x - a) % modulus
    if ctrl == 1:
        y = (y - b) % modulus
        if f1 == 1:
            lam = lam_r
            f1 = 0
        else:
            inverse = montgomery_inverse_value(int(x), width, modulus)
            lam = cast(int, montgomery_product_values(int(y), inverse, width, modulus))
    else:
        lam = 0
    return f1, ctrl, a, b, x, y, lam, lam_r


def addition_coordinates_values(
    modulus: int, ctrl: int, a: int, b: int, x: int, y: int, lam: int
) -> tuple[int, int, int, int, int, int]:
    """Retain the controlled coordinate overwrite and source THRU slope."""
    if ctrl == 1:
        x = (x + 3 * a) % modulus
        y = 0
    return ctrl, a, b, x, y, lam


def addition_products_values(
    width: int, modulus: int, x: int, y: int, lam: int
) -> tuple[int, int, int]:
    """Use the source square, subtraction and positive-slope product branch."""
    square = montgomery_product_values(int(lam), int(lam), width, modulus)
    x = (x - cast(int, square)) % modulus
    if lam > 0:
        y = cast(int, montgomery_product_values(int(x), int(lam), width, modulus))
    return x, y, lam


def addition_cleanup_values(
    modulus: int, ctrl: int, a: int, b: int, x: int, y: int, lam_r: int, lam: int
) -> tuple[int, int, int, int, int, int]:
    """Consume the slope logically; this supplies no physical erasure proof."""
    if ctrl == 1:
        x = (a - x) % modulus
        y = (y - b) % modulus
    else:
        x = (x + a) % modulus
    return ctrl, a, b, x, y, lam_r


def addition_finish_values(
    f1: int,
    f2: int,
    f3: int,
    f4: int,
    ctrl: int,
    a: int,
    b: int,
    x: int,
    y: int,
) -> tuple[int, int, int, int]:
    """Consume source flags with the literal infinity/opposite-point overrides."""
    if f4 == 1:
        x = a
        y = b
    if f1 and f2:
        x = 0
        y = 0
    return a, b, x, y


def elliptic_curve_addition_values(
    width: int, modulus: int, a: int, b: int, x: int, y: int, lam_r: int
) -> tuple[int, int, int, int, int]:
    """Evaluate the source affine callback independently of gate-width admission.

    Inputs/outputs are raw stored Montgomery words. The tangent determines the
    coefficient. Numerical callbacks carry no workspace, gates or cleanup claim.
    """
    tangent, left_y, left_x = (
        cast(int, montgomery_to_integer_value(value, width, modulus))
        for value in (lam_r, b, a)
    )
    curve_a = (tangent * 2 * left_y - 3 * left_x**2) % modulus
    right_x, right_y = (
        cast(int, montgomery_to_integer_value(value, width, modulus))
        for value in (x, y)
    )
    result_x, result_y = affine_point_add_values(
        (left_x, left_y), (right_x, right_y), modulus, curve_a
    )
    return (
        a,
        b,
        cast(int, integer_to_montgomery_value(result_x, width, modulus)),
        cast(int, integer_to_montgomery_value(result_y, width, modulus)),
        lam_r,
    )
