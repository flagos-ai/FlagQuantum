# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpec.is_active.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Evaluate bound classical control clauses without foreign dtype objects."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .value_parameters import is_symbolic_value


def classical_controls_active(
    values: Sequence[object],
    patterns: Sequence[NDArray[Any]],
    *,
    symbolic_parameters: Sequence[object] = (),
) -> bool:
    """Retain symbolic-first guards, exact shapes and first-mismatch short circuit.

    Patterns are already-bound activation arrays. The metadata owner supplies
    native width/shape parameters for the original dtype-symbolic guard; this
    function does not construct CtrlSpec or normalize its input patterns.
    """
    if is_symbolic_value(*symbolic_parameters) or is_symbolic_value(*patterns):
        raise ValueError("Cannot compute activation for symbolic control metadata")
    if len(values) != len(patterns):
        raise ValueError("Control values must match the number of bound patterns")
    for value, pattern in zip(values, patterns, strict=False):
        if isinstance(value, (int, np.integer)):
            array = np.array(value)
        else:
            array = np.asarray(value)
        if array.shape != pattern.shape:
            raise ValueError("Control value shape differs from its activation pattern")
        if np.any(array != pattern):
            return False
    return True
