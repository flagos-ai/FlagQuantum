# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/polynomial/qsp_testing.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/polynomial/qsp_testing.py

"""Random QSP coefficient generation and sampled GQSP pair consistency."""

from collections.abc import Sequence

import numpy as np
from numpy.polynomial import Polynomial


def random_qsp_polynomial(
    degree: int,
    *,
    random_state: np.random.RandomState,
    only_real_coeffs: bool = False,
) -> Sequence[complex]:
    """Return the source's degree coefficients, of maximum degree degree-1.

    Draw degree uniform magnitudes divided by degree and, unless real-only,
    draw degree uniform phases. Preserve source RNG consumption, empty degree
    zero output, and NumPy's negative-size refusal. The coefficient l1 norm
    bounds the polynomial on the circle; no quantum operation is synthesized.
    """
    coefficients = random_state.random(size=degree) / degree
    if not only_real_coeffs:
        coefficients = coefficients * np.exp(
            random_state.random(size=degree) * np.pi * 2j
        )
    return list(coefficients)


def check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
    left: Sequence[complex],
    right: Sequence[complex],
    *,
    random_state: np.random.RandomState,
    rtol: float = 1e-7,
    n_points: int = 1000,
) -> None:
    """Assert abs(left(z))**2 + abs(right(z))**2 ~= 1 at sampled points.

    Construct both NumPy polynomials before consuming a single uniform draw
    of n_points angles. Retain the source allclose defaults, failure, and RNG
    advancement. This finite sample check is not an all-circle certificate.
    No measurements, gates, work wires, or physical cost model are produced.
    """
    left_polynomial = Polynomial(left)
    right_polynomial = Polynomial(right)
    points = np.exp(random_state.random(size=n_points) * np.pi * 2j)
    np.testing.assert_allclose(
        np.abs(left_polynomial(points)) ** 2 + np.abs(right_polynomial(points)) ** 2,
        1,
        rtol=rtol,
    )
