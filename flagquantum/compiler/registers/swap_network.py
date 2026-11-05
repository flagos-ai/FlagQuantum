# Adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs
# Changes: native IR for the two-bank SwapWithZero specialization.

"""Two-bank SwapWithZero using Qualtran's relative-phase controlled swap."""

from ...core.ir import Instruction


def swap_with_zero_pair_ir(
    control: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Emit CSwapApprox and its MultiTargetCNOT in upstream gate order.

    When enabled, the sign is negative for odd parity of the bitwise overlap
    between the input words. A consumer erasing the words must undo this sign.
    """
    if not left or len(left) != len(right):
        raise ValueError("SwapWithZero requires equally sized nonempty banks")
    left_to_right = tuple(
        Instruction("cx", pair) for pair in zip(left, right, strict=True)
    )
    right_to_left = tuple(
        Instruction("cx", pair) for pair in zip(right, left, strict=True)
    )
    g = tuple(
        Instruction(name, (wire,))
        for wire in right
        for name in ("s", "h", "t", "h", "sdg")
    )
    g_inverse = tuple(
        Instruction(name, (wire,))
        for wire in right
        for name in ("s", "h", "tdg", "h", "sdg")
    )

    fanout_layers = tuple(
        tuple(
            Instruction("cx", (right[index], right[index + stride]))
            for index in range(min(stride, len(right) - stride))
        )
        for stride in (1 << level for level in range((len(right) - 1).bit_length()))
    )
    return (
        *right_to_left,
        *g_inverse,
        *left_to_right,
        *g_inverse,
        *(gate for layer in reversed(fanout_layers) for gate in layer),
        Instruction("cx", (control, right[0])),
        *(gate for layer in fanout_layers for gate in layer),
        *g,
        *left_to_right,
        *g,
        *right_to_left,
    )
