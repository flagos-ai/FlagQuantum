"""Recompute the original seeded matrix without silently applying its unused shift."""

import math

import numpy as np
import torch

rng = np.random.RandomState(1235)
raw = rng.rand(8, 8)
MATRIX = torch.tensor((raw + raw.T) / 2, dtype=torch.float64)
EIGENVALUES, _eigenvectors = torch.linalg.eigh(MATRIX)
maximum = float(EIGENVALUES.abs().max())
UNUSED_SHIFTED_MATRIX = (MATRIX + maximum) / (2 * maximum)
# The source computes a shifted matrix but exponentiates MATRIX. Preserve that
# actual calculation; interpreting the spectrum as lying in [0,1) would be wrong.
# Evaluate the Hermitian exponential in its orthonormal eigenbasis. This keeps
# its Gram residual within the strict unitary admission bound across BLAS builds.
_unitary = (
    _eigenvectors.to(torch.complex128) * torch.exp(2j * math.pi * EIGENVALUES)
) @ _eigenvectors.mT.to(torch.complex128)
UNITARY = tuple(tuple(complex(v) for v in row) for row in _unitary.tolist())
PRECISIONS = tuple(range(1, 9))
HISTORICAL_QISKIT_DEPTHS = (320, 950, 2206, 4706, 9694, 19658, 39574, 79394)
HISTORICAL_QISKIT_CX = (162, 484, 1124, 2398, 4938, 10008, 20136, 40378)
