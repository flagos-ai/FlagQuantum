# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix simd/b64_polyfill.h, b128_polyfill.h,
# b128_sse.h, b256_polyfill.h and b512_polyfill.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Unsigned packed integer operations over independent physical scalar lanes."""

from collections.abc import Iterable, Iterator


def _lane_mask(lane_bits: int) -> int:
    if type(lane_bits) is not int or lane_bits not in (32, 64):
        raise ValueError("Physical scalar lanes require 32 or 64 bits")
    return (1 << lane_bits) - 1


def word_mask(width: int) -> int:
    if type(width) is not int or width <= 0 or width % 32:
        raise ValueError("Packed width must be a positive multiple of 32")
    return (1 << width) - 1


def _check_word(word: int, width: int) -> None:
    maximum = word_mask(width)
    if type(word) is not int or not 0 <= word <= maximum:
        raise ValueError("Physical packed word must fit its unsigned width")


def pack_lanes(values: Iterable[int], lane_bits: int) -> int:
    """Pack low-to-high unsigned lanes without changing their scalar values."""
    mask = _lane_mask(lane_bits)
    result = 0
    for index, value in enumerate(values):
        if type(value) is not int or not 0 <= value <= mask:
            raise ValueError("Physical scalar value must fit its unsigned lane")
        result |= value << (lane_bits * index)
    return result


def unpack_lanes(word: int, width: int, lane_bits: int) -> tuple[int, ...]:
    mask = _lane_mask(lane_bits)
    _check_word(word, width)
    if width % lane_bits:
        raise ValueError("Packed width must contain an integral number of scalar lanes")
    return tuple((word >> offset) & mask for offset in range(0, width, lane_bits))


def broadcast_u32(value: int, width: int) -> int:
    """Repeat a uint32 value in every scalar lane of a packed word."""
    word_mask(width)
    return pack_lanes((value for _ in range(width // 32)), 32)


def add_lanes(left: int, right: int, width: int, lane_bits: int) -> int:
    """Add with independent scalar wraparound and no carry between lanes."""
    a = unpack_lanes(left, width, lane_bits)
    b = unpack_lanes(right, width, lane_bits)
    mask = (1 << lane_bits) - 1
    return pack_lanes(((x + y) & mask for x, y in zip(a, b, strict=True)), lane_bits)


def shift_lanes(
    word: int, amount: int, width: int, lane_bits: int, *, left: bool = False
) -> int:
    """Shift independent lanes, with zero for counts beyond the lane width.

    Intrinsic counts narrow to uint32 before the lane-width zero rule.
    Portable C++ shifts are compared only below their scalar width.
    """
    lanes = unpack_lanes(word, width, lane_bits)
    if type(amount) is not int or not 0 <= amount < 1 << 64:
        raise ValueError("Physical shift count must fit unsigned size_t")
    amount &= (1 << 32) - 1
    if amount >= lane_bits:
        return 0
    mask = (1 << lane_bits) - 1
    if left:
        return pack_lanes(((value << amount) & mask for value in lanes), lane_bits)
    return pack_lanes((value >> amount for value in lanes), lane_bits)


def equal_u32(left: int, right: int, width: int) -> int:
    """Return all-one uint32 lanes exactly where the scalar values are equal."""
    a = unpack_lanes(left, width, 32)
    b = unpack_lanes(right, width, 32)
    mask = (1 << 32) - 1
    return pack_lanes((mask * (x == y) for x, y in zip(a, b, strict=True)), 32)


def set_word_bit(word: int, index: int, value: bool, width: int) -> int:
    _check_word(word, width)
    if type(index) is not int or not 0 <= index < width:
        raise IndexError("Physical packed bit index is out of range")
    if type(value) is not bool:
        raise ValueError("Physical packed bit assignment requires a Boolean")
    mask = 1 << index
    return (word & ~mask) | (int(value) << index)


def random_word(
    width: int, draws: Iterator[int], *, high_lane_first: bool = False
) -> int:
    """Consume one uint64 draw per lane in the requested source call order.

    Portable source structs initialize low lanes first. The observed Clang SSE
    intrinsic constructor evaluates its high-lane argument first.
    """
    word_mask(width)
    if width % 64:
        raise ValueError("Random packed width must contain integral uint64 lanes")
    values = [next(draws) for _ in range(width // 64)]
    return pack_lanes(reversed(values) if high_lane_first else values, 64)


def format_word(word: int, width: int) -> str:
    """Render source low-bit-first underscores and ones."""
    _check_word(word, width)
    return f"{word:0{width}b}"[::-1].replace("0", "_")
