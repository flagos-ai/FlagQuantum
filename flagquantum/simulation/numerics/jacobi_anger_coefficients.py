# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/polynomial/jacobi_anger_approximations.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/polynomial/jacobi_anger_approximations.py

"""Laurent coefficients with optional caller-owned Bessel evaluation."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def approx_exp_cos_by_jacobi_anger(
    t: float,
    *,
    degree: int,
    bessel_jn: Callable[[NDArray[np.int64], float], NDArray[np.float64]] | None = None,
) -> NDArray[np.complex128]:
    """Return i**n * J_n(t), ordered n=-degree,...,degree.

    Evaluating the resulting ascending polynomial at exp(i theta) requires the
    Laurent offset -degree to approximate exp(i t cos(theta)). The provider is
    called once with the complete order array, as in the source. This numerical
    inspection creates no circuit or physical resource estimate. If omitted, the
    evaluator is SciPy's lazily imported jv; import/provider errors propagate.
    """
    orders = np.arange(-degree, degree + 1)
    evaluator = bessel_jn
    if evaluator is None:
        from scipy.special import jv

        evaluator = jv
    result: NDArray[np.complex128] = 1j**orders * evaluator(orders, t)
    return result


def approx_exp_sin_by_jacobi_anger(
    t: float,
    *,
    degree: int,
    bessel_jn: Callable[[NDArray[np.int64], float], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Return real J_n(t), ordered n=-degree,...,degree.

    The Laurent offset -degree approximates exp(i t sin(theta)). Retain the
    source's actual real array dtype despite its complex return annotation.
    A supplied provider owns evaluation and accuracy. With no provider, use
    lazily imported SciPy jv. Source failures and precision are not repaired.
    """
    orders = np.arange(-degree, degree + 1)
    evaluator = bessel_jn
    if evaluator is None:
        from scipy.special import jv

        evaluator = jv
    return evaluator(orders, t)
