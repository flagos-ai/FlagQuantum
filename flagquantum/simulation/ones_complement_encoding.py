# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _IntOnesComp and QIntOnesComp/CIntOnesComp.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Source ones' complement value domain, diagnostics and scalar/array codecs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ..core.ir.program.register_types import validate_ones_complement_width
from .integer_encoding import integer_from_bits, integer_to_bits
from .value_encoding import decode_value_array, encode_value_array

if TYPE_CHECKING:
    from sympy import Expr


def ones_complement_domain(width: int | Expr) -> range:
    """Retain the original domain, excluding both unsigned-magnitude endpoints."""
    validate_ones_complement_width(width)
    maximum = 1 << (width - 1)
    return range(-maximum + 1, maximum)


def validate_ones_complement_value(value: object, width: int | Expr) -> None:
    """Retain source inclusive bounds, even though encoding rejects the endpoints."""
    validate_ones_complement_width(width)
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"val should be an integer, not {value!r}")
    maximum = 1 << (width - 1)
    if not -maximum <= value <= maximum:
        raise ValueError(
            f"Classical value {value} must be in range [-{maximum}, +{maximum}] in val"
        )


def validate_ones_complement_array(values: NDArray[Any], width: int | Expr) -> None:
    """Retain the source default's flattened scalar checks rather than range-only arrays."""
    validate_ones_complement_width(width)
    for value in values.reshape(-1):
        validate_ones_complement_value(value, width)


def ones_complement_to_bits(value: object, width: int | Expr) -> list[int]:
    """Encode magnitude via the shared unsigned scalar codec and XOR with the sign."""
    validate_ones_complement_value(value, width)
    x = cast(int, value)
    return [int(x < 0)] + [
        bit ^ int(x < 0) for bit in integer_to_bits(abs(x), width - 1)
    ]


def ones_complement_from_bits(bits: Sequence[int], width: int | Expr) -> int | float:
    """Decode suffix XOR sign before applying its sign power, with no length repair."""
    validate_ones_complement_width(width)
    magnitude = integer_from_bits([bit ^ bits[0] for bit in bits[1:]], width)
    return cast(int | float, (-1) ** int(bits[0]) * magnitude)


def ones_complement_array_to_bits(
    values: NDArray[Any], width: int | Expr
) -> NDArray[np.uint8]:
    """Use the original default vectorized scalar encoder and inference."""
    validate_ones_complement_width(width)
    return encode_value_array(
        values, lambda value: ones_complement_to_bits(value, width)
    )


def ones_complement_array_from_bits(
    bits: NDArray[Any], width: int | Expr
) -> NDArray[Any]:
    """Keep scalar negative-zero interpretation and source vectorized result types."""
    validate_ones_complement_width(width)
    return decode_value_array(
        bits, lambda row: ones_complement_from_bits(cast(Sequence[int], row), width)
    )
