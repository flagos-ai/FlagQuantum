# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Allocate, Free, Split, Join and Cast value callbacks.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping

"""Compose numerical scalar codecs without foreign register or graph objects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sympy import Expr

Input = TypeVar("Input")
Output = TypeVar("Output")


def allocate_classical_value(
    width: int | Expr, decode: Callable[[Sequence[int]], Output]
) -> Output:
    """Decode the original zero list; dirty borrowing does not change this callback."""
    return decode([0] * cast(int, width))


def free_classical_value(value: object, *, dirty: bool = False) -> dict[str, object]:
    """Check value before dirty permission, then consume the numerical register.

    This does not release memory or apply a quantum reset. Ambiguous array
    comparisons still fail even when dirty permission is true, as in the source.
    """
    if value != 0 and not dirty:
        raise ValueError(
            "Cannot free a nonzero classical register without dirty permission"
        )
    return {}


def split_classical_value(
    value: Input, encode: Callable[[Input], Sequence[int]]
) -> NDArray[Any]:
    """Encode then apply source np.asarray without forcing a NumPy bit dtype."""
    return np.asarray(encode(value))


def join_classical_value(
    bits: NDArray[Any], decode: Callable[[Sequence[int]], Output]
) -> Output:
    """Call tolist before scalar decoding, preserving the original conversion order."""
    return decode(bits.tolist())


def cast_classical_value(
    value: Input,
    encode: Callable[[Input], Sequence[int]],
    decode: Callable[[Sequence[int]], Output],
) -> Output:
    """Pass the exact encoded object to the output codec, without array conversion.

    These callbacks consume already bound codec/width metadata. Compiler owns
    width and quantum/classical layout legality; codecs own numerical domains.
    """
    return decode(encode(value))
