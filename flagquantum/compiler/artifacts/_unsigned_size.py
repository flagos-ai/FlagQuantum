# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Integer protocols follow the locked builder allocation and table-width casts.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit

"""Convert Python integer protocols to bounded physical storage sizes."""

from typing import SupportsIndex, SupportsInt


def unsigned_size(value: object) -> int:
    """Normalize an integer result without invoking an int subclass override."""
    if isinstance(value, int):
        result = int.__int__(value)
    elif isinstance(value, (SupportsInt, SupportsIndex)) and not isinstance(
        value, float
    ):
        result = int(value)
    else:
        raise TypeError("Physical storage size requires an integer")
    if not 0 <= result < 1 << 64:
        raise TypeError("Physical storage size must fit unsigned 64-bit storage")
    return result
