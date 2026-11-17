# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _fkn_matrix and TwoBitFFFT.my_tensors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/two_bit_ffft.py

"""Analytical fermionic Fourier transform in logical occupation order."""

import numpy as np
from numpy.typing import NDArray


def two_bit_fermionic_fourier_matrix(
    k: int, n: int, *, adjoint: bool = False
) -> NDArray[np.complex128]:
    """Return the source tensor's matrix before the circuit's global phase."""
    if type(k) is not int or type(n) is not int or n == 0:
        raise ValueError("Fermionic Fourier transform requires integer k and nonzero n")
    if type(adjoint) is not bool:
        raise ValueError("Fermionic Fourier transform direction must be boolean")
    factor = 1 / np.sqrt(2)
    phase = np.exp(2j * np.pi * k / n)
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, factor, phase * factor, 0],
            [0, factor, -phase * factor, 0],
            [0, 0, 0, -phase],
        ],
        dtype=np.complex128,
    )
    return matrix.conj().T if adjoint else matrix
