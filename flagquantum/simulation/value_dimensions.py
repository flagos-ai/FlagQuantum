# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Shaped, HasLength, slen and shape.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/types.py

"""Shape-only numerical values, independent of register signatures and graph IR."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from numpy.typing import NDArray

if TYPE_CHECKING:
    from sympy import Expr


@dataclass(frozen=True, slots=True)
class SymbolicArrayShape:
    """An unknown array with the original tuple-only shape validation.

    Concrete dimensions do not imply known values: this descriptor is always
    symbolic. Dimensions are not coerced or individually validated here.
    """

    shape: tuple[int | Expr, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("A symbolic numerical array shape must be a tuple")

    def is_symbolic(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class SymbolicSequenceLength:
    """An unknown sequence retaining the configured length without coercion.

    Like the source, do not implement __len__: Python cannot return symbolic
    expressions or negative diagnostic lengths from that protocol.
    """

    length: int | Expr

    def is_symbolic(self) -> bool:
        return True


ArrayValue: TypeAlias = NDArray[Any] | SymbolicArrayShape


def value_length(
    value: Sized | SymbolicArrayShape | SymbolicSequenceLength,
) -> int | Expr:
    """Use array's first dimension, configured sequence length or ordinary len."""
    if isinstance(value, SymbolicArrayShape):
        return value.shape[0]
    if isinstance(value, SymbolicSequenceLength):
        return value.length
    return len(value)


def value_shape(value: ArrayValue) -> tuple[int | Expr, ...]:
    """Return the original shape object without interpreting its dimensions."""
    return value.shape
