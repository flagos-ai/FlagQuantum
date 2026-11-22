# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/word_ops.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Unsigned 64-bit limb operations shared by declared-width scalar values."""

WORD_MASK = (1 << 64) - 1


def require_word(*values: int) -> None:
    if any(type(value) is not int or not 0 <= value <= WORD_MASK for value in values):
        raise ValueError("Limb values must be unsigned 64-bit integers")


def add_carry(carry: bool, left: int, right: int) -> tuple[int, bool]:
    """Return the low sum word and its outgoing carry."""
    require_word(left, right)
    total = left + right + carry
    return total & WORD_MASK, bool(total >> 64)


def subtract_borrow(borrow: bool, left: int, right: int) -> tuple[int, bool]:
    """Return the low difference word and its outgoing borrow."""
    require_word(left, right)
    difference = left - right - borrow
    return difference & WORD_MASK, difference < 0


def extract_bits(value: int, mask: int) -> int:
    """Pack selected bits into low lanes in increasing source-bit order."""
    require_word(value, mask)
    result, output_bit = 0, 1
    while mask:
        input_bit = mask & -mask
        if value & input_bit:
            result |= output_bit
        mask &= mask - 1
        output_bit <<= 1
    return result


def full_multiply_add(
    left: int, right: int, offset1: int, offset2: int
) -> tuple[int, int]:
    """Return low and high limbs of the exact double-offset product."""
    require_word(left, right, offset1, offset2)
    total = left * right + offset1 + offset2
    return total & WORD_MASK, total >> 64
