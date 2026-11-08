# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SignTruncate construction and on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/sign_extension.py

"""Checked signed numerical truncation through shared big-endian codecs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..core.value_parameters import is_symbolic_value
from .classical_bitwise import IntegerValue
from .integer_encoding import integer_from_bits, integer_to_bits
from .ones_complement_encoding import (
    ones_complement_from_bits,
    ones_complement_to_bits,
    validate_ones_complement_width,
)

if TYPE_CHECKING:
    from sympy import Expr


def signed_truncate_classical_value(
    value: IntegerValue,
    input_width: int | Expr,
    output_width: int | Expr,
    *,
    ones_complement: bool = False,
) -> int | float:
    """Keep original constructor order, value cast, prefix checks and raw decode.

    Width indices are not normalized to integers. The output is a numerical
    value; raw negative-zero bit patterns remain the quantum circuit's concern.
    """
    if ones_complement:
        validate_ones_complement_width(input_width)
        validate_ones_complement_width(output_width)
    drop = input_width - output_width
    if not is_symbolic_value(drop) and drop <= 0:
        raise ValueError("Input signed width must be larger than output width")
    if ones_complement:
        bits = ones_complement_to_bits(int(value), input_width)
    else:
        bits = integer_to_bits(int(value), input_width, signed=True)
    removed = bits[: cast(int, drop)]
    sign = int(bits[cast(int, drop)])
    if any(bit != sign for bit in removed):
        raise ValueError(f"Removed prefix {removed} must equal sign bit {sign}")
    suffix = bits[cast(int, drop) :]
    if ones_complement:
        return ones_complement_from_bits(suffix, output_width)
    return integer_from_bits(suffix, output_width, signed=True)
