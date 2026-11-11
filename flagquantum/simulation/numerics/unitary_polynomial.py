# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/polynomial/basic.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/polynomial/basic.py

"""Dense polynomial inspection of a supplied unitary, without circuit synthesis."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_Matrix = (
    NDArray[np.float32]
    | NDArray[np.float64]
    | NDArray[np.complex64]
    | NDArray[np.complex128]
)


def evaluate_polynomial_of_unitary_matrix(
    coefficients: Sequence[complex],
    unitary: _Matrix,
    *,
    offset: int = 0,
) -> _Matrix:
    """Evaluate U**offset * sum(coefficients[i] * U**i) in source order.

    Coefficients are in ascending degree order. Negative offsets use powers of
    the conjugate transpose, relying on caller-supplied unitarity, not an inverse
    or a numerical unitarity check. Square shape is asserted as in the source.
    Output retains U's dtype; incompatible coefficient casts retain NumPy errors.
    Inputs are not mutated. Empty coefficients return a zero matrix.

    This NumPy inspection routine stores O(N**2) dense data and performs one
    matrix product per coefficient, in addition to the initial matrix power.
    It generates no gates, work wires, measurements, or resource estimates and
    does not expose a differentiable torch execution path.
    """
    assert unitary.ndim == 2 and unitary.shape[0] == unitary.shape[1]
    if offset < 0:
        power = np.linalg.matrix_power(unitary.conj().T, -offset)
    else:
        power = np.linalg.matrix_power(unitary, offset)
    result = np.zeros(unitary.shape, dtype=unitary.dtype)
    for coefficient in coefficients:
        np.add(result, power * coefficient, out=result)
        power = power @ unitary
    return result
