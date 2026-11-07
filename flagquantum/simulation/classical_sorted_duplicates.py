# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HasDuplicates.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/has_duplicates.py

"""Sorted duplicate value callback with original dictionary and mutation semantics."""

from collections.abc import Mapping

import numpy as np

from .classical_bitwise import IntegerValue


def sorted_duplicates_classical_values(
    registers: Mapping[str, IntegerValue],
) -> dict[str, IntegerValue]:
    """Preserve extra fields and source augmented XOR, including ndarray mutation.

    The source callback does not inspect controlled-mode metadata or ctrl.
    Controlled quantum execution has a different documented activation contract.
    """
    values = dict(registers)
    xs = np.asarray(values["xs"])
    assert np.all(xs == np.sort(xs))
    if np.any(xs[:-1] == xs[1:]):
        values["flag"] ^= 1
    return values
