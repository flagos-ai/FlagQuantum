# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the pinned Split, Join and integer encodings in Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Ordered register bindings and big-endian integer bit interpretation."""

from ..bits.bitwise import _validate_wires


def split_register(packed: tuple[int, ...]) -> tuple[tuple[int], ...]:
    """Bind each physical wire as a one-bit port without emitting a gate."""
    if packed:
        _validate_wires(packed)
    return tuple((wire,) for wire in packed)


def join_register(bits: tuple[tuple[int], ...]) -> tuple[int, ...]:
    """Bind ordered one-bit ports as one flat physical register."""
    if any(len(bit) != 1 for bit in bits):
        raise ValueError("Register join requires one wire per bit")
    packed = tuple(bit[0] for bit in bits)
    if packed:
        _validate_wires(packed)
    return packed


def split_integer(value: int, width: int, *, signed: bool = False) -> tuple[int, ...]:
    """Encode a concrete integer, with two's complement for signed words."""
    if type(width) is not int or width < 1:
        raise ValueError("Integer encoding requires a positive concrete width")
    limit = 1 << (width - 1 if signed else width)
    if type(value) is not int or not (-limit if signed else 0) <= value < limit:
        raise ValueError("Integer value must fit the register encoding")
    return tuple((value >> shift) & 1 for shift in reversed(range(width)))


def join_integer(bits: tuple[int, ...], *, signed: bool = False) -> int:
    """Decode a nonempty big-endian word in the selected integer encoding."""
    if not bits or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("Integer decoding requires a nonempty binary word")
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value - (1 << len(bits)) if signed and bits[0] else value
