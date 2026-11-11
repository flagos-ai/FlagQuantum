# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/testing.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/testing.py

"""Distinct phase-sensitive and phase-insensitive dense comparison assertions."""

import numpy as np
from numpy.typing import NDArray

from .unitary_distance import unitary_distance_ignoring_global_phase

_Matrix = NDArray[np.float32 | np.float64 | np.complex64 | np.complex128]


def assert_matrices_almost_equal(
    left: _Matrix, right: _Matrix, *, atol: float = 1e-5
) -> None:
    """Assert equal shapes, then the source default norm of the difference <= atol.

    For matrices this is the Frobenius norm, not a spectral or entrywise bound.
    Global phase is retained. No broadcasting, unitarity check, normalization,
    tolerance repair, physical operation, or resource estimate is added.
    """
    assert left.shape == right.shape
    assert np.linalg.norm(left - right) <= atol


def assert_unitaries_equivalent_upto_global_phase(
    left: _Matrix, right: _Matrix, *, atol: float = 1e-5
) -> None:
    """Assert the shared phase-insensitive unitary distance <= atol.

    Reuse the source trace metric and its shape errors and roundoff clamp.
    Unitarity remains the caller's responsibility. This cannot replace full
    complex-amplitude or controlled-operation conformance. Numerical comparison
    uses dense O(N**2) storage and generates no physical instructions.
    """
    distance = unitary_distance_ignoring_global_phase(left, right)
    assert (
        distance <= atol
    ), f"unitaries are not equivalent: distance={distance} ({atol=})"
