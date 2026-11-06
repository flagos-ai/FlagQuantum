# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the concrete qubit mapping of Qualtran v0.7.0 Partition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py

"""Ordered concrete register partitioning without quantum operations."""

from .bitwise import _validate_wires

RegisterLayout = tuple[tuple[str, int], ...]


def partition_register(
    packed: tuple[int, ...], layout: RegisterLayout
) -> dict[str, tuple[int, ...]]:
    """Bind flat big-endian qubits to the ordered named subregisters."""
    parts: dict[str, tuple[int, ...]] = {}
    offset = 0
    for name, width in layout:
        if not name or name in parts:
            raise ValueError("Register partition requires distinct nonempty names")
        if type(width) is not int or width < 0:
            raise ValueError("Register partition requires nonnegative integer widths")
        parts[name] = packed[offset : offset + width]
        offset += width
    if offset != len(packed):
        raise ValueError("Packed register width must equal the sum of partition widths")
    if packed:
        _validate_wires(packed)
    return parts
