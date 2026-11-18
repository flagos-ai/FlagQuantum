# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 qualtran/bloqs/qsp/fast_qsp.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Flip-convolution complementary QSP optimization with caller-owned randomness."""

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray


class ComplementaryPolynomialObjective:
    """Retain the source real/complex convolution objective and coefficient order.

    The real mode requires float64 coefficients and the complex mode requires
    complex128 coefficients. Optimizer vectors pack all real coordinates before
    all imaginary coordinates. Input boundedness and convergence are caller-owned.
    """

    def __init__(
        self,
        poly: NDArray[np.float64] | NDArray[np.complex128],
        only_reals: bool = False,
    ) -> None:
        self.only_reals = only_reals
        if self.only_reals:
            assert poly.dtype == np.float64
            self.conv_p_negative = self.conv_by_flip_conj(poly) * -1
        else:
            assert poly.dtype == np.complex128
            self.conv_p_negative = (
                self.complex_conv_by_flip_conj(poly.real, poly.imag) * -1
            )
        self.conv_p_negative[poly.shape[0] - 1] = 1 - np.linalg.norm(poly) ** 2

    def loss_function(self, x: NDArray[np.float64]) -> float:
        """Return squared convolution distance, retaining the source norm order."""
        if self.only_reals:
            conv_result = self.conv_by_flip_conj(x)
        else:
            real_part = x[: len(x) // 2]
            imag_part = x[len(x) // 2 :]
            conv_result = self.complex_conv_by_flip_conj(real_part, imag_part)
        return float(np.linalg.norm(self.conv_p_negative - conv_result) ** 2)

    @staticmethod
    def array_to_complex(x: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Unpack consecutive real and imaginary coefficient halves."""
        real_part = x[: len(x) // 2]
        imag_part = x[len(x) // 2 :]
        return real_part + 1.0j * imag_part

    @staticmethod
    def conv_by_flip_conj(
        poly: NDArray[np.float64] | NDArray[np.complex128],
    ) -> NDArray[np.float64] | NDArray[np.complex128]:
        """Evaluate the source flip convolution used by the real mode."""
        return np.convolve(poly, np.flip(poly, axis=(0,)), mode="full")

    @staticmethod
    def complex_conv_by_flip_conj(
        real_part: NDArray[np.float64], imag_part: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Evaluate split-complex conjugate convolution in the source order."""
        real_flip = np.flip(real_part, axis=(0,))
        imag_flip = np.flip(-1 * imag_part, axis=(0,))
        conv_real_part = np.convolve(real_part, real_flip, mode="full")
        conv_imag_part = np.convolve(imag_part, imag_flip, mode="full")
        conv_real_imag = np.convolve(real_part, imag_flip, mode="full")
        conv_imag_real = np.convolve(imag_part, real_flip, mode="full")
        real_conv = conv_real_part - conv_imag_part
        imag_conv = conv_real_imag + conv_imag_real
        return real_conv + 1j * imag_conv


def fast_complementary_polynomial(
    coefficients: Sequence[float] | Sequence[complex],
    random_state: np.random.RandomState,
    only_reals: bool = False,
    tolerance: float = 1e-10,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Optimize an ascending complementary polynomial using Equation 60.

    SciPy is imported when called. The explicit RNG supplies a normalized Gaussian
    initial vector. Preserve the source three-point finite difference, tolerance,
    unconstrained optimization and returned iterate, including unsuccessful exits.
    A returned array alone therefore does not certify complementarity. Real mode
    retains its lower numerical accuracy and original coercion failures. Dense
    convolutions require linear storage and quadratic arithmetic per evaluation.

    Examples:
        >>> p = (0.2, 0.3)
        >>> q = fast_complementary_polynomial(p, np.random.RandomState(42))
        >>> len(q) == len(p)
        True
    """
    from scipy.optimize import minimize

    if only_reals:
        poly = np.array(coefficients, dtype=np.float64)
        q_initial = random_state.randn(poly.shape[0])
    else:
        poly = np.array(coefficients, dtype=np.complex128)
        q_initial = random_state.randn(poly.shape[0] * 2)
    q_initial_normalized = q_initial / np.linalg.norm(q_initial)
    objective = ComplementaryPolynomialObjective(poly, only_reals=only_reals)
    minimizer = minimize(
        objective.loss_function, q_initial_normalized, jac="3-point", tol=tolerance
    )
    result = cast(NDArray[np.float64], minimizer.x)
    if only_reals:
        return result
    return objective.array_to_complex(result)
