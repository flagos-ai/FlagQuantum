# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Swap, CSwap, CSwapApprox and SwapWithZero.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/cswap_approx.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/swap_with_zero.py

"""Raw exchange callbacks, separate from relative-phase quantum execution."""

from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .classical_bitwise import IntegerValue

Left = TypeVar("Left")
Right = TypeVar("Right")


def swap_classical_values(left: Left, right: Right) -> tuple[Right, Left]:
    """Exchange the original objects without conversions, copies or width checks."""
    return right, left


def controlled_swap_classical_values(
    control: IntegerValue, left: Left, right: Right
) -> tuple[int, Left | Right, Left | Right]:
    """Compare control with zero then one and return a literal integer control.

    This is the shared raw callback for exact and relative-phase exchanges.
    It describes the basis permutation only; CSwapApprox quantum phases remain
    the responsibility of its existing Compiler decomposition.
    """
    if control == 0:
        return 0, left, right
    if control == 1:
        return 1, right, left
    raise ValueError("Bad control value for controlled exchange classical simulation")


def swap_with_zero_classical_values(
    stages: Iterable[tuple[int, int, int, int]],
    row_shape: tuple[int, ...],
    *,
    targets: NDArray[Any],
    **selection: IntegerValue,
) -> dict[str, IntegerValue | NDArray[Any]]:
    """Apply the stage order in place and return the original selection values."""
    assert isinstance(targets, np.ndarray)
    values = tuple(selection.values())
    for axis, bit, left, right in stages:
        if values[axis] & (1 << bit):
            left_index = np.unravel_index(left, row_shape)
            right_index = np.unravel_index(right, row_shape)
            targets[left_index], targets[right_index] = (
                targets[right_index],
                targets[left_index],
            )
    return {**selection, "targets": targets}
