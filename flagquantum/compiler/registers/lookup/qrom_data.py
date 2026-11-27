# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 data_loading/qrom.py.

"""Compile-time raw equality and visited-cell loading shared by QROM protocols."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray


def qrom_break_early(
    data: tuple[NDArray[Any], ...],
    prefix: tuple[int, ...],
    start: int,
    stop: int,
) -> bool:
    """Compare all raw cells in the source slice, including ignored columns."""
    for array in data:
        values = array[prefix][start:stop].flat
        first = values[0]
        if any(value != first for value in values):
            return False
    return True


def qrom_load_words(
    data: tuple[NDArray[Any], ...],
    target_shapes: tuple[tuple[int, ...], ...],
    prefix: tuple[int, ...],
) -> Iterator[tuple[int, tuple[int, ...], int]]:
    """Convert only source-visited declared target cells to integer XOR words."""
    for dataset, (array, shape) in enumerate(zip(data, target_shapes, strict=True)):
        for index in np.ndindex(shape):
            yield dataset, index, int(array[prefix + index])
