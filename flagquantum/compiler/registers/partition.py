# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the concrete and symbolic register mapping of Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py

"""Ordered register partitioning without quantum operations."""

from __future__ import annotations

from collections.abc import Mapping

from .._component_trace.symbolic_math import SymbolicInt
from .bitwise import _validate_wires
from .register_bits import join_integer, split_integer

RegisterLayout = tuple[tuple[str, int], ...]


def partition_width(layout: tuple[tuple[str, SymbolicInt], ...]) -> SymbolicInt:
    """Count logical bits for a concrete or symbolic bookkeeping partition."""
    names: set[str] = set()
    total: SymbolicInt = 0
    for name, width in layout:
        if not name or name in names:
            raise ValueError("Register partition requires distinct nonempty names")
        if isinstance(width, (int, float)) and (type(width) is not int or width < 0):
            raise ValueError("Register partition requires nonnegative integer widths")
        names.add(name)
        total += width
    return total


def partition_register(
    packed: tuple[int, ...], layout: RegisterLayout
) -> dict[str, tuple[int, ...]]:
    """Bind flat big-endian qubits to the ordered named subregisters."""
    if any(type(width) is not int for _, width in layout):
        raise ValueError("Physical partition requires concrete integer widths")
    if partition_width(layout) != len(packed):
        raise ValueError("Packed register width must equal the sum of partition widths")
    if packed:
        _validate_wires(packed)
    parts: dict[str, tuple[int, ...]] = {}
    offset = 0
    for name, width in layout:
        parts[name] = packed[offset : offset + width]
        offset += width
    return parts


def unpartition_register(
    parts: Mapping[str, tuple[int, ...]], layout: RegisterLayout
) -> tuple[int, ...]:
    """Flatten named ports in declared layout order without emitting a gate."""
    if any(type(width) is not int for _, width in layout):
        raise ValueError("Physical partition requires concrete integer widths")
    partition_width(layout)
    if parts.keys() != {name for name, _ in layout}:
        raise ValueError("Register join requires exactly the declared port names")
    if any(len(parts[name]) != width for name, width in layout):
        raise ValueError("Register join port widths must match the layout")
    packed = tuple(wire for name, _ in layout for wire in parts[name])
    if packed:
        _validate_wires(packed)
    return packed


def split_register_pair(
    packed: tuple[int, ...], left_width: int, right_width: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Bind the high and low words of a two-port partition."""
    parts = partition_register(packed, (("y1", left_width), ("y2", right_width)))
    return parts["y1"], parts["y2"]


def join_register_pair(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Bind high and low physical words as one packed register."""
    return unpartition_register(
        {"y1": left, "y2": right}, (("y1", len(left)), ("y2", len(right)))
    )


def split_integer_pair(
    value: int, left_width: int, right_width: int
) -> tuple[int, int]:
    """Split an unsigned packed integer into two nonempty unsigned words."""
    if any(type(width) is not int or width < 1 for width in (left_width, right_width)):
        raise ValueError("Integer split requires two positive concrete widths")
    bits = split_integer(value, left_width + right_width)
    return join_integer(bits[:left_width]), join_integer(bits[left_width:])


def _join_word(value: int, width: int) -> int:
    if type(width) is not int or width < 0:
        raise ValueError("Integer join requires nonnegative concrete widths")
    if type(value) is not int or not 0 <= value < 1 << max(width, 64):
        raise ValueError("Integer join value exceeds the source word range")
    # The pinned array encoder masks machine words and range-checks wider ones.
    return value & ((1 << width) - 1)


def join_integer_pair(left: int, right: int, left_width: int, right_width: int) -> int:
    """Pack the two unsigned words using the source array encoding rules."""
    high, low = _join_word(left, left_width), _join_word(right, right_width)
    if left_width + right_width == 0:
        raise ValueError("Integer join requires a nonempty result")
    return (high << right_width) | low
