# Copyright 2024 Google LLC
# Copyright 2018 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/matrix.py::random_hermitian_matrix
# and Cirq 1.7.0 testing/lin_alg_utils.py::random_orthogonal.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/matrix.py

"""Source-order real symmetric matrix sampling from an explicit NumPy RNG."""

import numpy as np
from numpy.typing import NDArray


def random_hermitian_matrix(
    dimension: int, random_state: np.random.RandomState
) -> NDArray[np.float64]:
    """Sample the source's real symmetric Hermitian matrix with norm <= 1.

    Consume one dimension-by-dimension normal draw from the caller-owned RNG,
    perform QR, correct column signs, then average with the conjugate transpose.
    This reproduces the source's real orthogonal sampling distribution, not a
    general complex Hermitian ensemble. RNG state advances even if QR fails.

    The native boundary takes an explicit RandomState, as annotated upstream;
    Cirq's additional seed/None parsing is not provided. NumPy dimension errors
    and degenerate QR division behavior are retained without retries or repairs.
    Storage is O(dimension**2); QR is O(dimension**3). No gates, work wires,
    resource model, device selection, or differentiable execution are produced.
    """
    matrix: NDArray[np.float64] = random_state.randn(dimension, dimension)
    orthogonal, triangular = np.linalg.qr(matrix)
    diagonal = np.diag(triangular)
    orthogonal = orthogonal * (diagonal / abs(diagonal))
    result: NDArray[np.float64] = (orthogonal + orthogonal.conj().T) / 2
    return result
