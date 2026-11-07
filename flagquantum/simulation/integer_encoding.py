# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitEncoding, _UInt and _Int value codecs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Original big-endian scalar and NumPy integer value encodings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .integer_values import validate_integer_word_value

if TYPE_CHECKING:
    from sympy import Expr


def _symbolic_width(width: int | Expr) -> bool:
    if isinstance(width, (int, np.integer)):
        return False
    from sympy import Basic

    if isinstance(width, Basic):
        return True
    checker = getattr(width, "is_symbolic", None)
    return False if checker is None else bool(checker())


def integer_to_bits(
    value: object, width: int | Expr, *, signed: bool = False
) -> list[int]:
    """Encode using source range checks, unsigned formatting or signed binary_repr.

    Signed encoding rejects symbolic widths, including Sympy integer literals.
    Unsigned scalar formatting retains its original symbolic-literal behavior.
    No dtype casts precede the source range checks.
    """
    if signed and _symbolic_width(width):
        raise ValueError("Cannot compute signed bits with a symbolic width")
    validate_integer_word_value(value, width, signed=signed)
    if signed:
        return [
            int(bit) for bit in np.binary_repr(cast(int, value), width=cast(int, width))
        ]
    return [int(bit) for bit in f"{int(cast(int, value)):0{width}b}"]


def integer_from_bits(
    bits: Sequence[int], width: int | Expr, *, signed: bool = False
) -> int:
    """Decode with the original unchecked scalar bit-string interpretation.

    Unsigned decoding ignores width. Signed decoding reads the first bit and
    complements the suffix, with the original width-one special case. This
    diagnostic method does not certify binary elements or matching bit length.
    """
    if not signed:
        return int("".join(str(bit) for bit in bits), 2)
    sign = bits[0]
    value = (
        0
        if width == 1
        else integer_from_bits(
            [1 - bit if sign else bit for bit in bits[1:]], width - 1
        )
    )
    return ~value if sign else value


def integer_array_to_bits(
    values: NDArray[Any], width: int | Expr, *, signed: bool = False
) -> NDArray[np.uint8]:
    """Retain source vectorization and the <=64-bit unsigned mask/transpose path.

    This is not a generalized replacement broadcasting policy. Source failures
    for incompatible array shapes, dtypes, symbolic widths and empty vectorized
    inputs propagate. The fast unsigned path does not add scalar range checks.
    """
    if signed:
        return cast(
            "NDArray[np.uint8]",
            np.vectorize(
                lambda x: np.asarray(
                    integer_to_bits(x, width, signed=True), dtype=np.uint8
                ),
                signature="()->(n)",
            )(values),
        )
    if _symbolic_width(width):
        raise ValueError("Cannot compute unsigned array bits with a symbolic width")
    if width > 64:
        return cast(
            "NDArray[np.uint8]",
            np.vectorize(
                lambda x: np.asarray(integer_to_bits(x, width), dtype=np.uint8),
                signature="()->(n)",
            )(values),
        )
    w = int(width)
    array = np.atleast_1d(values)
    if not np.issubdtype(array.dtype, np.uint):
        if not np.all(array >= 0):
            raise AssertionError
        if not np.iinfo(array.dtype).bits <= 64:
            raise AssertionError
        array = array.astype(np.uint64)
    if not w <= np.iinfo(array.dtype).bits:
        raise AssertionError
    mask = 2 ** np.arange(w - 1, -1, -1, dtype=array.dtype).reshape((w, 1))
    return cast("NDArray[np.uint8]", (array & mask).astype(bool).astype(np.uint8).T)


def integer_array_from_bits(
    bits: NDArray[Any], width: int | Expr, *, signed: bool = False
) -> NDArray[Any]:
    """Retain source array reduction and vectorize result inference above 64 bits."""
    if signed:
        return cast(
            "NDArray[Any]",
            np.vectorize(
                lambda row: integer_from_bits(row, width, signed=True),
                signature="(n)->()",
            )(bits),
        )
    bitstrings = np.atleast_2d(bits)
    if bitstrings.shape[1] != width:
        raise ValueError(
            f"Input bitsize {bitstrings.shape[1]} does not match width={width}"
        )
    if width > 64:
        return cast(
            "NDArray[Any]",
            np.vectorize(
                lambda row: integer_from_bits(row, width), signature="(n)->()"
            )(bits),
        )
    basis = 2 ** np.arange(width - 1, -1, -1, dtype=np.uint64)
    return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)
