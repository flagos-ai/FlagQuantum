# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SingleWindowModMul.on_classical_vals and qrom data.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Source single-window value action with low-first controls and big-endian target."""

from __future__ import annotations

from collections.abc import Sequence
from math import log2
from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue
from .classical_montgomery_multiplication import _inversion_data
from .integer_encoding import integer_from_bits, integer_to_bits
from .value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr


def montgomery_window_bits_classical_values(
    left: Sequence[int],
    right: IntegerValue,
    target: Sequence[int],
    index: IntegerValue,
    width: int | Expr,
    window: int | Expr,
    modulus: int | Expr,
) -> dict[str, object]:
    """Port the bit-array window method without graph objects or gate execution.

    The source treats left[0] as the least-significant control, but target is
    big endian. Incoming index is overwritten, not XORed. Input sequences and
    right retain identity; output target is the source list of Python bits.
    The raw table and metadata failures are evaluated at their original stage.
    Clean quantum execution separately requires a zero initial index and the
    compiler's valid layout/domain, not arbitrary values admitted by this method.
    """
    symbolic_shape = is_symbolic_value(width) or is_symbolic_value(window)
    if not symbolic_shape:
        if not width % window == 0:
            raise AssertionError
    if symbolic_shape:
        raise ValueError(
            "Classical bit-window action does not support symbolic dimensions"
        )
    n, w = cast(int, width), cast(int, window)
    target_width = n + w
    value: IntegerValue = integer_from_bits(target, target_width)
    for i in range(w):
        if left[i]:
            value += right << i
    index = value & (2**w - 1)
    if is_symbolic_value(modulus):
        # Original SingleWindow passes log_block_sizes=None explicitly to a
        # QROAM tuple converter. Preserve that construction failure before
        # metadata validation rather than silently choosing a block size.
        raise TypeError("'NoneType' object is not iterable")
    data = _inversion_data(w, cast(int, modulus))
    # The source QROAM constructor validates its integer size metadata after
    # creating the dataset and selecting a block. No block allocation is needed
    # for a numerical value, but retain its nonpositive-size arithmetic failures.
    if target_width <= 0:
        log2(2**w / target_width)
    if not isinstance(w, int) or not isinstance(target_width, int):
        raise AssertionError
    correction = data[index]
    value = (value + correction) >> w
    result = integer_to_bits(value, target_width)
    return {"target": result, "qrom_index": index, "x": left, "y": right}
