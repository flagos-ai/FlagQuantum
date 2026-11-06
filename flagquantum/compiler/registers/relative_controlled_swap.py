# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwapApprox.decompose_from_registers.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/cswap_approx.py

"""Controlled exchange with the source's relative signs."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, fanout_ir


def relative_controlled_swap_ir(
    control: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Exchange equal-width words under control, retaining source relative phases.

    The fanout conjugation is self-adjoint and uses four T gates per pair.
    """
    if not left or len(left) != len(right):
        raise ValueError("Relative controlled swap requires equal nonempty words")
    _validate_wires((control, *left, *right))
    right_to_left = tuple(
        Instruction("cx", (y, x)) for x, y in zip(left, right, strict=True)
    )
    left_to_right = tuple(
        Instruction("cx", (x, y)) for x, y in zip(left, right, strict=True)
    )

    def rotate(phase: str) -> tuple[Instruction, ...]:
        return tuple(
            Instruction(gate, (wire,))
            for wire in right
            for gate in ("s", "h", phase, "h", "sdg")
        )

    inverse_rotation, rotation = rotate("tdg"), rotate("t")
    return (
        *right_to_left,
        *inverse_rotation,
        *left_to_right,
        *inverse_rotation,
        *fanout_ir(control, right),
        *rotation,
        *left_to_right,
        *rotation,
        *right_to_left,
    )
