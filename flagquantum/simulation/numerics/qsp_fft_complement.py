# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 qualtran/bloqs/qsp/fft_qsp.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""FFT complementary QSP coefficients in the source scaling and mode order."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

_Scalar = TypeVar("_Scalar", np.float64, np.complex128)


def fft_complementary_polynomial(
    coefficients: Sequence[float] | Sequence[complex],
    tolerance: float = 1e-4,
    num_modes: int = 500,
) -> NDArray[np.complex128]:
    """Compute the source approximate complement using Fourier multiplication.

    Coefficients have ascending degree. Preserve scaling by 1-tolerance/4,
    padding by num_modes-1, forward-normalized transforms and the multiplier's
    cutoff based on num_modes. The transform length is len(coefficients)+num_modes-1.
    This finite approximation assumes bounded input and retains invalid-log,
    padding and transform failures. Tolerance supplies a scale margin, not a
    certified whole-circle residual bound. Temporary storage is linear in the
    transform length, with FFT arithmetic O(n log n). Inputs remain caller-owned.

    Examples:
        >>> q = fft_complementary_polynomial((0.2, 0.3), tolerance=1e-9)
        >>> q.shape
        (2,)
    """
    poly = np.array(coefficients)

    def scale(
        x: NDArray[_Scalar],
    ) -> NDArray[_Scalar]:
        return (1 - tolerance / 4) * x

    def pad_poly(
        x: NDArray[_Scalar],
    ) -> NDArray[_Scalar]:
        return np.pad(scale(x), (0, num_modes - 1))

    def p_eval(
        x: NDArray[_Scalar],
    ) -> NDArray[np.complex128]:
        return np.fft.ifft(pad_poly(x), norm="forward")

    def get_log(
        x: NDArray[_Scalar],
    ) -> NDArray[np.float64]:
        return np.log(1 - (np.abs(p_eval(x))) ** 2)

    def fourier_multiplier(
        the_log: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        modes = np.fft.fft(the_log, norm="forward")
        modes[0] *= 1 / 2
        modes[num_modes // 2 + 1 :] = 0
        return modes

    def get_modes(
        x: NDArray[_Scalar],
    ) -> NDArray[np.complex128]:
        return np.fft.ifft(fourier_multiplier(get_log(x)), norm="forward")

    def calculate_coeff(
        x: NDArray[_Scalar],
    ) -> NDArray[np.complex128]:
        return np.fft.fft(np.exp(get_modes(x)), norm="forward")[: x.shape[0]]

    return calculate_coeff(poly)
