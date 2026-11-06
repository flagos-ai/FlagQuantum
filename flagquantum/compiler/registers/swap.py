# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitSwap and Swap.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Literal swaps of equally sized registers."""

from __future__ import annotations

from ...core.ir import Instruction
from .._component_trace.symbolic_math import SymbolicInt
from .bitwise import _validate_wires
from .symbolic_resource_model import SymbolicResourceFormula


def swap_bit_ir(left: int, right: int) -> tuple[Instruction, ...]:
    """Emit the source atomic SWAP on two distinct wires."""
    return swap_register_ir((left,), (right,))


def swap_register_ir(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Exchange corresponding pairs in declared order, with no auxiliary wires."""
    if not left or len(left) != len(right):
        raise ValueError("Swap requires equal nonempty register widths")
    _validate_wires((*left, *right))
    return tuple(Instruction("swap", (x, y)) for x, y in zip(left, right, strict=True))


def source_swap_resource_formula(width: SymbolicInt) -> SymbolicResourceFormula:
    """Count one abstract Clifford per pair and retain the original wire width."""
    if isinstance(width, (int, float)) and (type(width) is not int or width < 1):
        raise ValueError("Swap resources require a positive integer width")
    return SymbolicResourceFormula(0, width, 0, 2 * width)
