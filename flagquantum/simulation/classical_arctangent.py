# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ArcTan.apply.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arctan.py

"""Own the original binary64 normalized arctangent, without external codecs."""

import numpy as np


def normalized_arctangent_value(value: int) -> np.float64:
    """Evaluate -2*atan(value)/pi with the original ufunc and range assertion."""
    output = np.float64(-2 * np.arctan(value, dtype=np.double) / np.pi)
    assert -1 <= output <= 1
    return output
