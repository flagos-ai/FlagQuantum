# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/polynomial/jacobi_anger_approximations.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/polynomial/jacobi_anger_approximations.py

"""Laurent coefficients with caller-owned integer-order Bessel evaluation."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def approx_exp_cos_by_jacobi_anger(
    t: float,
    *,
    degree: int,
    bessel_jn: Callable[[NDArray[np.int64], float], NDArray[np.float64]],
) -> NDArray[np.complex128]:
    """Return i**n * J_n(t), ordered n=-degree,...,degree.

    Evaluating the resulting ascending polynomial at exp(i theta) requires the
    Laurent offset -degree to approximate exp(i t cos(theta)). The provider is
    called once with the complete order array, as in the source. This numerical
    inspection creates no circuit or physical resource estimate.
    """
    orders = np.arange(-degree, degree + 1)
    result: NDArray[np.complex128] = np.multiply(1j**orders, bessel_jn(orders, t))
    return result


def approx_exp_sin_by_jacobi_anger(
    t: float,
    *,
    degree: int,
    bessel_jn: Callable[[NDArray[np.int64], float], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return real J_n(t), ordered n=-degree,...,degree.

    The Laurent offset -degree approximates exp(i t sin(theta)). Retain the
    source's actual real array dtype despite its complex return annotation.
    The caller owns Bessel evaluation, its failures, and approximation precision.
    """
    orders = np.arange(-degree, degree + 1)
    return bessel_jn(orders, t)
