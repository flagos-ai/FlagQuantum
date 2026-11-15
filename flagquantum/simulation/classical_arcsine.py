# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ArcSin.on_classical_vals, arithmetic/trigonometric/arcsin.py.

"""The locked ArcSin scalar word semantics, separate from reversible arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr


def source_arcsine_classical_values(
    x: int, result: int, width: int | Expr
) -> tuple[int, int]:
    """XOR the original full-width fractional arcsine into a scalar result.

    The source uses the full width as the fractional precision, even if its
    signature declares a different split. Its output is an unmasked integer.
    Integer inputs that round outside [0, 1) at the binary64 conversion
    retain the source assertion.
    """
    if is_symbolic_value(width):
        raise ValueError(f"Symbolic bitsize {width} not supported")
    scale = 2**width
    fraction = x / scale
    assert 0 <= fraction < 1
    # The source's unsigned fixed-point conversion truncates at this scale.
    quantized = int(fraction * scale) / scale
    return x, result ^ int(np.arcsin(quantized) * scale)
