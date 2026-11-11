# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/polynomial/qsp_testing.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/polynomial/qsp_testing.py

"""Finite FFT-grid QSP inspection, without a whole-circle bound guarantee."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

_NumericArray = NDArray[np.float64 | np.complex128 | np.int64]
_Coefficients = Sequence[complex] | _NumericArray


def polynomial_max_abs_value_on_unit_circle(
    coefficients: _Coefficients,
    *,
    fft: Callable[[_NumericArray], NDArray[np.complex128]],
    n_points: int = 2**17,
) -> float:
    """Return the source maximum on n_points uniform FFT-grid points.

    Zero-pad coefficients without changing dtype and call the supplied FFT once.
    The explicit provider replaces the source's SciPy import. Retain assignment,
    FFT and reduction failures rather than truncating or resizing invalid input.
    Numeric storage is O(n_points); no gates or quantum resources are generated.
    """
    data: _NumericArray = np.asarray(coefficients)
    padded: _NumericArray = np.zeros(n_points, dtype=data.dtype)
    padded[: len(data)] = data
    values = fft(padded)
    return float(np.max(np.abs(values)))


def scale_down_to_qsp_polynomial(
    coefficients: _Coefficients,
    *,
    fft: Callable[[_NumericArray], NDArray[np.complex128]],
    n_points: int = 2**17,
) -> _NumericArray:
    """Divide by the sampled maximum only when it exceeds one.

    Match the original asarray/list/padding sequence and dtype behavior. Return
    the same array when no scaling is needed; scaled results allocate a new array.
    This source heuristic can miss peaks between grid points and cannot certify
    a QSP polynomial over the entire circle.
    """
    data: _NumericArray = np.asarray(coefficients)
    maximum = polynomial_max_abs_value_on_unit_circle(
        list(data), fft=fft, n_points=n_points
    )
    if maximum > 1:
        data = np.divide(data, maximum)
    return data


def assert_is_qsp_polynomial(
    coefficients: _Coefficients,
    *,
    fft: Callable[[_NumericArray], NDArray[np.complex128]],
    n_points: int = 2**17,
) -> None:
    """Retain the source <=1 assertion for its finite-grid maximum only."""
    maximum = polynomial_max_abs_value_on_unit_circle(
        coefficients, fft=fft, n_points=n_points
    )
    assert (
        maximum <= 1
    ), f"Not a QSP polynomial! maximum absolute value {maximum} is greater than 1."
