# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 DirtyOutOfPlaceMontgomeryModMul value methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Source Montgomery integer windows, history accumulation and checked value erasure."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def _parameters(
    width: int | Expr, window: int | Expr, modulus: int | Expr
) -> tuple[int, int, int]:
    # Preserve the source constructor's Python-int guards and their order.
    if isinstance(modulus, int):
        if not (modulus > 1 and modulus % 2 == 1):
            raise AssertionError
    if isinstance(modulus, int) and isinstance(width, int):
        if not 2 * modulus - 1 < 2**width:
            raise AssertionError(f"bitsize={width} is too small")
    if isinstance(window, int) and isinstance(width, int):
        if not width % window == 0:
            raise AssertionError
    if not all(isinstance(value, int) for value in (width, window, modulus)):
        from sympy import Expr

        if any(isinstance(value, Expr) for value in (width, window, modulus)):
            raise ValueError(
                "Classical Montgomery action does not support symbolic parameters"
            )
    # Type narrowing only: do not coerce NumPy or symbolic values to Python int.
    return cast(int, width), cast(int, window), cast(int, modulus)


def _inversion_data(window: int, modulus: int) -> NDArray[np.integer[Any]]:
    inverse = pow(modulus, 2 ** (window - 1) - 1, 2**window)
    size = 2**window
    data = (-np.arange(size) * inverse) % size
    data *= modulus
    return cast("NDArray[np.integer[Any]]", data)


def _window_values(
    left: IntegerValue,
    right: IntegerValue,
    target: IntegerValue,
    history: IntegerValue,
    window: int,
    data: Callable[[], NDArray[np.integer[Any]]],
) -> tuple[IntegerValue, IntegerValue]:
    for i in range(window):
        if (left >> i) & 1:
            target += right << i
    index = target & (2**window - 1)
    correction = data()[index]
    target += correction
    target >>= window
    history = (history << window) | index
    return target, history


def montgomery_window_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    target: IntegerValue,
    history: IntegerValue,
    width: int | Expr,
    window: int | Expr,
    modulus: int | Expr,
) -> tuple[IntegerValue, IntegerValue]:
    """Port the original integer window method, retaining NumPy table arithmetic.

    Parameters follow the source enclosing multiplication constructor. Values
    are not cast or normalized. Low bits of left are consumed least significant
    first; the pre-correction target index is appended to history big endian.
    """
    _, window, modulus = _parameters(width, window, modulus)
    return _window_values(
        left, right, target, history, window, lambda: _inversion_data(window, modulus)
    )


def montgomery_multiply_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    width: int | Expr,
    window: int | Expr,
    modulus: int | Expr,
    *,
    target: IntegerValue | None = None,
    history: IntegerValue | None = None,
    reduced: IntegerValue | None = None,
    uncompute: bool = False,
) -> dict[str, IntegerValue]:
    """Port full dirty Montgomery value evaluation with original assertion boundaries.

    Forward inputs strictly inside (0,mod) use the original windows and one
    final conditional reduction. Exterior inputs return three Python zeros.
    Reverse checks the original Fermat-based target expression and presence of
    history/reduction, but does not validate their contents. This is not a proof
    that arbitrary history can be erased by the quantum adjoint.
    """
    width, window, modulus = _parameters(width, window, modulus)
    if uncompute:
        if not (
            target is not None
            and target
            == (left * right * pow(2, width * (modulus - 2), modulus)) % modulus
        ):
            raise AssertionError
        if history is None:
            raise AssertionError
        if reduced is None:
            raise AssertionError
        return {"x": left, "y": right}
    if target is not None:
        raise AssertionError
    if history is not None:
        raise AssertionError
    if reduced is not None:
        raise AssertionError
    if not (0 < left < modulus and 0 < right < modulus):
        return {"x": left, "y": right, "target": 0, "qrom_indices": 0, "reduced": 0}

    target = 0
    history = 0
    reduced = 0
    cached: NDArray[np.integer[Any]] | None = None

    def data() -> NDArray[np.integer[Any]]:
        nonlocal cached
        if cached is None:
            cached = _inversion_data(window, modulus)
        return cached

    for i in range(0, width, window):
        target, history = _window_values(
            left >> i, right, target, history, window, data
        )
    if target >= modulus:
        target -= modulus
        reduced = 1
    expected = (left * right * pow(2, width * (modulus - 2), modulus)) % modulus
    if not target == expected:
        raise AssertionError
    return {
        "x": left,
        "y": right,
        "target": target,
        "qrom_indices": history,
        "reduced": reduced,
    }
