# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/hubbard_model/qubitization/prepare_hubbard.py.

"""Evaluate Hubbard preparation coefficients without owning circuit construction."""

from math import sqrt
from typing import overload

from ...core.value_math import SymbolicReal


@overload
def hubbard_normalization(dimension: int, t: float, u: float) -> float: ...


@overload
def hubbard_normalization(
    dimension: int, t: SymbolicReal, u: SymbolicReal
) -> SymbolicReal: ...


def hubbard_normalization(
    dimension: int, t: SymbolicReal, u: SymbolicReal
) -> SymbolicReal:
    """Retain source integer arithmetic and the interaction contribution's floor."""
    modes = 2 * dimension**2
    return 2 * modes * t + (modes * u) // 2


def hubbard_conditional_preparation_angle() -> float:
    """Retain the fixed source NumPy angle shared by both coefficient paths."""
    import numpy as np

    return float(2 * np.arccos(np.sqrt(1 / 5)))


def hubbard_preparation_angles(
    dimension: int, t: float, u: float
) -> tuple[float, float]:
    """Retain numerator order, math sqrt, and NumPy arccos/conditional sqrt.

    This numerical helper preserves original warnings and arithmetic failures.
    Callers own their distinct physical-operation and interchange validators.
    """
    import numpy as np

    modes = 2 * dimension**2
    return (
        float(2 * np.arccos(sqrt(t * modes / hubbard_normalization(dimension, t, u)))),
        hubbard_conditional_preparation_angle(),
    )
