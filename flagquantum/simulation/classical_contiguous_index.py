# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ToContiguousIndex.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/contiguous_index.py

"""Original raw triangular index arithmetic, separate from finite-width XOR gates."""

from typing import cast

from .classical_bitwise import IntegerValue


def contiguous_index_classical_values(
    mu: IntegerValue, nu: IntegerValue
) -> tuple[IntegerValue, IntegerValue, IntegerValue]:
    """Preserve inputs and evaluate the source polynomial without casts or wrapping."""
    return mu, nu, cast(IntegerValue, nu * (nu + 1) // 2 + mu)
