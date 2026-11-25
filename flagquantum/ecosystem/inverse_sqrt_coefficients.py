# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/inverse_sqrt.py.

"""Convert supplied inverse-root coefficients through the original real codec."""

from collections.abc import Sequence

import numpy as np

from .fixed_point import fixed_point_real_to_integer


def inverse_sqrt_coefficient_qrom_data(
    selection_width: int,
    target_width: int,
    coefficients: tuple[Sequence[float], Sequence[float]],
) -> tuple[tuple[int, ...], ...]:
    """Retain original repeated addresses, wrapping and platform-integer failures.

    The two coefficient sequences are zipped as in the original helper. Output
    has four banks, including zero at address zero. No alternate quantizer or
    precision fallback is selected when the optional real codec fails.
    """
    low, high = coefficients
    data = np.zeros((4, 2**selection_width), dtype=np.int_)
    for row, (a, b) in enumerate(zip(low, high, strict=False)):
        data[row, 1] = fixed_point_real_to_integer(a, target_width, target_width)
        data[row, 2] = fixed_point_real_to_integer(b, target_width, target_width)
        data[row, 3] = fixed_point_real_to_integer(
            a / 2 ** (1 / 2), target_width, target_width
        )
        start = 4
        for exponent in range(2, selection_width):
            count = max(1, 2 ** (exponent - 1))
            end = start + count
            data[row, start:end] = fixed_point_real_to_integer(
                a / 2 ** (exponent / 2), target_width, target_width
            )
            start += count
            end += count
            data[row, start:end] = fixed_point_real_to_integer(
                b / 2 ** (exponent / 2), target_width, target_width
            )
            start = end
    return tuple(tuple(int(word) for word in bank) for bank in data)
