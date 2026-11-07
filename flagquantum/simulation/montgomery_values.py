# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _MontgomeryUInt numeric methods and domain.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Original Montgomery representation arithmetic, separate from gate algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue
from .integer_encoding import _symbolic_width

if TYPE_CHECKING:
    from sympy import Expr


def montgomery_value_domain(width: int | Expr, modulus: int | Expr | None) -> range:
    """Enumerate source nonzero residues, or all stored words for an absent/symbolic mod."""
    if modulus is None or _symbolic_width(modulus):
        return range(cast(int, 2**width))
    return range(1, int(modulus))


def _modulus(modulus: int | Expr | None) -> int:
    if modulus is None or _symbolic_width(modulus):
        raise AssertionError
    # Narrow only; the source casts the modulus to int at selected pow calls,
    # but not at the inverse's first pow or the final remainder.
    return cast(int, modulus)


def integer_to_montgomery_value(
    value: IntegerValue, width: int | Expr, modulus: int | Expr | None
) -> IntegerValue | Expr:
    """Retain x * pow(2, int(width), int(mod)) before remainder with original mod."""
    mod = _modulus(modulus)
    return cast("IntegerValue | Expr", (value * pow(2, int(width), int(mod))) % mod)


def montgomery_to_integer_value(
    value: IntegerValue, width: int | Expr, modulus: int | Expr | None
) -> IntegerValue | Expr:
    """Use the original inverse power-of-two factor without operand casts."""
    mod = _modulus(modulus)
    return cast("IntegerValue | Expr", (value * pow(2, -int(width), int(mod))) % mod)


def montgomery_product_values(
    left: IntegerValue,
    right: IntegerValue,
    width: int | Expr,
    modulus: int | Expr | None,
) -> IntegerValue | Expr:
    """Multiply both stored operands before the source inverse-factor multiplication.

    Do not reduce intermediates or convert NumPy operands to Python integers.
    The source imposes no extra scalar value/range validation in this method.
    """
    mod = _modulus(modulus)
    return cast(
        "IntegerValue | Expr", (left * right * pow(2, -int(width), int(mod))) % mod
    )


def montgomery_inverse_value(
    value: IntegerValue, width: int | Expr, modulus: int | Expr | None
) -> IntegerValue | Expr:
    """Retain the source's asymmetric casts in its two modular powers.

    The first pow receives the original operand and modulus. The second casts
    only modulus; unlike product/conversion it does not cast the width exponent.
    Noninvertibility and unsupported NumPy/symbolic pow inputs propagate.
    """
    mod = _modulus(modulus)
    return cast(
        "IntegerValue | Expr",
        (pow(cast(int, value), -1, mod) * pow(2, 2 * width, int(mod))) % mod,
    )
