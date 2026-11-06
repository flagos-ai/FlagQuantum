# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitCSwap.to_clifford_t_circuit and CSwap.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py
# Changes: explicit native operands, retaining the literal exact seven-T stream.

"""Exact controlled bit/register exchange without relative-phase substitution."""

from ...core.ir import Instruction
from .bitwise import _validate_wires


def controlled_swap_bit_ir(
    control: int, left: int, right: int
) -> tuple[Instruction, ...]:
    """Emit the source exact Fredkin Clifford+T circuit, with no work or measures."""
    _validate_wires((control, left, right))
    return (
        Instruction("cx", (right, left)),
        Instruction("cx", (control, left)),
        Instruction("h", (right,)),
        Instruction("t", (control,)),
        Instruction("tdg", (left,)),
        Instruction("t", (right,)),
        Instruction("cx", (right, left)),
        Instruction("cx", (control, right)),
        Instruction("t", (left,)),
        Instruction("cx", (control, left)),
        Instruction("tdg", (right,)),
        Instruction("tdg", (left,)),
        Instruction("cx", (control, right)),
        Instruction("cx", (right, left)),
        Instruction("t", (left,)),
        Instruction("h", (right,)),
        Instruction("cx", (right, left)),
    )


def controlled_swap_ir(
    control: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Exchange arbitrary equal-width words on a coherent positive control."""
    if not left or len(left) != len(right):
        raise ValueError("Controlled swap requires equal nonempty register widths")
    _validate_wires((control, *left, *right))
    return tuple(
        gate
        for x, y in zip(left, right, strict=True)
        for gate in controlled_swap_bit_ir(control, x, y)
    )
