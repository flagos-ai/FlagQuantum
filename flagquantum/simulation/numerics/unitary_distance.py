# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/matrix.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/matrix.py

"""The source's phase-insensitive dense unitary comparison metric."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def unitary_distance_ignoring_global_phase(
    left: NDArray[Any],
    right: NDArray[Any],
) -> np.float16 | np.float32 | np.float64 | np.longdouble:
    """Return sqrt(max(0, 1 - abs(trace(left.dagger @ right)) / N)).

    Inputs must have the same square shape; unitarity is the caller's
    responsibility, as in the pinned source. The clamp retains the source's
    roundoff behavior. This deliberately ignores global phase and therefore
    cannot certify exact complex amplitudes or controlled-operation equivalence.
    It uses O(N**2) dense storage and emits no physical instructions or costs.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Inputs must have same dimension, got {left.shape}, {right.shape}"
        )
    if left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError(f"Inputs must be square matrices, got shape {left.shape}")
    dimension = left.shape[0]
    trace = np.trace(left.conj().T @ right)
    distance_squared = 1 - np.abs(trace) / dimension
    result: np.float16 | np.float32 | np.float64 | np.longdouble = np.sqrt(
        max(0, distance_squared)
    )
    return result
