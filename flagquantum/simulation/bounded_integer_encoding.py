# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _BUInt scalar and default array codecs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Big-endian bounded integer codecs through shared numerical boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .bounded_integer_values import (
    validate_bounded_integer_shape,
    validate_bounded_integer_value,
)
from .integer_encoding import integer_from_bits, integer_to_bits
from .value_encoding import decode_value_array, encode_value_array

if TYPE_CHECKING:
    from sympy import Expr


def _encode(value: object, width: int | Expr, bound: int | Expr) -> list[int]:
    validate_bounded_integer_value(value, bound)
    return integer_to_bits(value, width)


def _decode(bits: Sequence[int], width: int | Expr, bound: int | Expr) -> int:
    value = integer_from_bits(bits, width)
    validate_bounded_integer_value(value, bound)
    return value


def bounded_integer_to_bits(
    value: object, width: int | Expr, bound: int | Expr
) -> list[int]:
    """Validate the bounded value before the shared unsigned encoding checks."""
    validate_bounded_integer_shape(width, bound)
    return _encode(value, width, bound)


def bounded_integer_from_bits(
    bits: Sequence[int], width: int | Expr, bound: int | Expr
) -> int:
    """Decode without bit-length repair, then validate the resulting bounded value."""
    validate_bounded_integer_shape(width, bound)
    return _decode(bits, width, bound)


def bounded_integer_array_to_bits(
    values: NDArray[Any], width: int | Expr, bound: int | Expr
) -> NDArray[np.uint8]:
    """Use default scalar vectorization, retaining bound and magnitude checks."""
    validate_bounded_integer_shape(width, bound)
    return encode_value_array(values, lambda value: _encode(value, width, bound))


def bounded_integer_array_from_bits(
    bits: NDArray[Any], width: int | Expr, bound: int | Expr
) -> NDArray[Any]:
    """Keep the original result inference and post-decode bounded-value check."""
    validate_bounded_integer_shape(width, bound)
    return decode_value_array(
        bits, lambda row: _decode(cast(Sequence[int], row), width, bound)
    )
