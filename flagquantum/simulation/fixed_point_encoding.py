# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _Fxp stored-integer methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Fixed-point raw integer codecs, independent of real quantization libraries."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .integer_encoding import integer_from_bits, integer_to_bits
from .integer_values import validate_integer_word_value
from .value_encoding import decode_value_array, encode_value_array
from .value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr


def validate_fixed_point_shape(
    width: int | Expr, fractional_width: int | Expr, *, signed: bool = False
) -> None:
    """Keep the source _Fxp constructor checks and symbolic short circuits."""
    if not is_symbolic_value(width) and width == 1 and signed:
        raise ValueError("bitsize must be > 1.")
    if not is_symbolic_value(width) and not is_symbolic_value(fractional_width):
        if signed and width == fractional_width:
            raise ValueError("num_frac must be less than bitsize if the Fxp is signed.")
        if width < fractional_width:
            raise ValueError("bitsize must be >= num_frac.")


def fixed_point_integer_width(
    width: int | Expr, fractional_width: int | Expr, *, signed: bool = False
) -> int | Expr:
    """Return the original width subtraction without normalizing its operands."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    return width - fractional_width


def fixed_point_domain(
    width: int | Expr, fractional_width: int | Expr, *, signed: bool = False
) -> Iterator[int]:
    """Yield the raw same-width integer domain, not real fixed-point values."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)

    def values() -> Iterator[int]:
        # Source shape construction is eager; integer-domain iteration is lazy.
        if signed:
            maximum = 1 << (width - 1)
            yield from range(-maximum, maximum)
        else:
            yield from range(2**width)

    return values()


def validate_fixed_point_value(
    value: object,
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> None:
    """Validate the stored integer through the existing scalar word owner."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    validate_integer_word_value(value, width, signed=bool(signed))


def validate_fixed_point_array(
    values: NDArray[Any],
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> None:
    """Retain flattened scalar validation, including empty-array behavior."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    for value in values.reshape(-1):
        validate_integer_word_value(value, width, signed=bool(signed))


def fixed_point_to_bits(
    value: object,
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> list[int]:
    """Encode the stored integer; fractional width does not rescale it."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    return integer_to_bits(value, width, signed=bool(signed))


def fixed_point_from_bits(
    bits: Sequence[int],
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> int:
    """Decode using the original unchecked same-width integer interpretation."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    return integer_from_bits(bits, width, signed=bool(signed))


def fixed_point_array_to_bits(
    values: NDArray[Any],
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> NDArray[np.uint8]:
    """Use source default scalar vectorization, not the unsigned mask fast path."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    return encode_value_array(
        values, lambda value: integer_to_bits(value, width, signed=bool(signed))
    )


def fixed_point_array_from_bits(
    bits: NDArray[Any],
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> NDArray[Any]:
    """Retain default array result inference and scalar decode diagnostics."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    return decode_value_array(
        bits,
        lambda row: integer_from_bits(
            cast(Sequence[int], row), width, signed=bool(signed)
        ),
    )


def fixed_point_real_value(
    value: int | float | Expr,
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> int | float | Expr:
    """Apply source x / 2**num_frac without raw-word range validation or casts."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    return value / 2**fractional_width
