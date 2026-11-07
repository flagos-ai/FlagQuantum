# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Swap, CSwap and CSwapApprox.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/cswap_approx.py

"""Raw exchange callbacks, separate from relative-phase quantum execution."""

from typing import TypeVar

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
