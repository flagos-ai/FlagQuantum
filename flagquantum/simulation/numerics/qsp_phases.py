# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 qsp_phase_factors and SU2RotationGate.rotation_matrix.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qsp/generalized_qsp.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/su2_rotation.py

"""Numerical QSP phase recursion independent of physical gate compilation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def qsp_phase_factors(
    left: Sequence[complex], right: Sequence[complex]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return theta, phi and initial lambda in the source's absolute-phase gauge.

    Retain the original reverse-degree recursion, 1e-10 safe-angle threshold,
    shape assertion and unequal-length ValueError. Inputs are not mutated.
    Complementarity is the caller's responsibility and is not checked here.
    Empty pairs retain source empty arrays and zero lambda. Numerical work
    stores the shrinking 2-by-n coefficients; no physical rotations or cost
    model are emitted and no Compiler dependency is required.
    """
    if len(left) != len(right):
        raise ValueError("Polynomials P and Q must have the same degree.")
    coefficients = np.array([left, right])
    count = coefficients.shape[1]
    theta = np.zeros(count)
    phi = np.zeros(count)
    initial_lambda = 0.0

    def safe_angle(value: complex) -> float:
        return 0.0 if np.isclose(value, 0, atol=1e-10) else float(np.angle(value))

    for degree in reversed(range(count)):
        assert coefficients.shape == (2, degree + 1)
        a, b = coefficients[:, degree]
        theta[degree] = np.arctan2(np.abs(b), np.abs(a))
        phi[degree] = (
            0 if np.isclose(np.abs(b), 0, atol=1e-10) else safe_angle(a * np.conj(b))
        )
        if degree == 0:
            initial_lambda = safe_angle(b)
        else:
            # Same arithmetic as the source rotation_matrix at lambda=shift=0.
            rotation = np.exp(1j * 0) * np.array(
                [
                    [
                        np.exp(1j * (0 + phi[degree])) * np.cos(theta[degree]),
                        np.exp(1j * phi[degree]) * np.sin(theta[degree]),
                    ],
                    [np.exp(1j * 0) * np.sin(theta[degree]), -np.cos(theta[degree])],
                ]
            )
            coefficients = rotation.conj().T @ coefficients
            coefficients = np.array(
                [coefficients[0][1 : degree + 1], coefficients[1][0:degree]]
            )
    return theta, phi, initial_lambda
