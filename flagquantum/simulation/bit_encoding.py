# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _Bit, QBit/CBit value encodings.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Source single-bit equality validation, scalar identity decode and arrays."""

from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .value_encoding import decode_value_array, encode_value_array


def bit_value_domain() -> Iterator[int]:
    """Yield the original zero then one domain."""
    yield from (0, 1)


def validate_bit_value(value: object) -> None:
    """Retain equality-to-zero-or-one semantics, without an integer-type guard."""
    if not (value == 0 or value == 1):
        raise ValueError(f"Bad bit value: {value} in val")


def validate_bit_array(values: NDArray[Any]) -> None:
    """Retain original elementwise equality and all reduction, including empty arrays."""
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Bad bit value array in val")


def bit_to_bits(value: object) -> list[int]:
    """Validate, then make the source's explicit int conversion."""
    validate_bit_value(value)
    return [int(cast(int, value))]


def bit_from_bits(bits: Sequence[object]) -> object:
    """Require one element, returning that exact object without value validation."""
    if not len(bits) == 1:
        raise AssertionError
    return bits[0]


def bit_array_to_bits(values: NDArray[Any]) -> NDArray[np.uint8]:
    """Use shared source default vectorization of the bit scalar encoder."""
    return encode_value_array(values, bit_to_bits)


def bit_array_from_bits(bits: NDArray[Any]) -> NDArray[Any]:
    """Preserve scalar identity decode and vectorized output dtype inference."""
    return decode_value_array(
        bits, lambda row: bit_from_bits(cast(Sequence[object], row))
    )
