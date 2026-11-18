# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 KaiserWindowState.kaiser_state_coeff at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Compute the exact periodic Kaiser coefficients used by phase windows."""

import numpy as np
from numpy.typing import NDArray

from ...core.value_math import SymbolicReal


def kaiser_window_coefficients(
    bitsize: int, alpha: SymbolicReal
) -> NDArray[np.float64]:
    """Normalize the source SciPy periodic window without changing beta or length.

    SciPy is loaded only when coefficients are requested. Its parameter errors,
    overflow warnings and resulting NaNs retain their original behavior.
    """
    from scipy.signal.windows import kaiser

    state = np.asarray(kaiser(2**bitsize, alpha * np.pi, sym=False), dtype=np.float64)
    return state / np.linalg.norm(state)
