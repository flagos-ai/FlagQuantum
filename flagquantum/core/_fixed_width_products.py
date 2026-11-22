# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc product loops.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Source-ordered scalar/word products including sequential target aliases."""

from ._word_operations import WORD_MASK, add_carry, full_multiply_add


def accumulate_product(
    target: int,
    width: int,
    left: int,
    left_width: int | None,
    right: int,
    right_width: int,
    *,
    left_alias: bool = False,
    right_alias: bool = False,
    shift: int = 0,
    subtract: bool = False,
) -> int:
    """Accumulate using original limb order; None denotes a scalar left factor."""
    words = (width + 63) // 64
    mask = (1 << width) - 1
    if subtract:
        target = ~target & mask
    if left_width is None:
        limit = words if shift else min(words, (right_width + 63) // 64)
        carry = 0
        for index in range(limit):
            offset = 64 * index - shift
            value = target if right_alias else right
            factor = (
                0
                if offset <= -64
                else (value >> offset if offset >= 0 else value << -offset) & WORD_MASK
            )
            low, carry = full_multiply_add(
                left, factor, carry, (target >> (64 * index)) & WORD_MASK
            )
            target = (target & ~(WORD_MASK << (64 * index))) | (low << (64 * index))
        index = limit
        while index < words and carry:
            low, overflow = add_carry(
                False, (target >> (64 * index)) & WORD_MASK, carry
            )
            target = (target & ~(WORD_MASK << (64 * index))) | (low << (64 * index))
            carry = int(overflow)
            index += 1
    else:
        for first in range(min(words, (left_width + 63) // 64)):
            carry = 0
            limit = min(words - first, (right_width + 63) // 64)
            for second in range(limit):
                a = ((target if left_alias else left) >> (64 * first)) & WORD_MASK
                b = ((target if right_alias else right) >> (64 * second)) & WORD_MASK
                position = 64 * (first + second)
                low, carry = full_multiply_add(
                    a, b, carry, (target >> position) & WORD_MASK
                )
                target = (target & ~(WORD_MASK << position)) | (low << position)
            index = first + limit
            while index < words and carry:
                position = 64 * index
                low, overflow = add_carry(
                    False, (target >> position) & WORD_MASK, carry
                )
                target = (target & ~(WORD_MASK << position)) | (low << position)
                carry = int(overflow)
                index += 1
    return (~target if subtract else target) & mask
