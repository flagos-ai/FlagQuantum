# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CNOT, Toffoli and MultiTargetCNOT.on_classical_vals.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs

"""Classical controlled-bit callbacks with their original distinct predicates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def cnot_classical_values(
    control: IntegerValue, target: IntegerValue
) -> tuple[IntegerValue, IntegerValue]:
    """Return the original addition-modulo-two callback without operand casts."""
    return control, (control + target) % 2


def toffoli_classical_values(
    controls: NDArray[np.integer[Any]], target: IntegerValue
) -> tuple[NDArray[np.integer[Any]], IntegerValue]:
    """Assert a binary target, then flip only when both entries equal one."""
    assert target in [0, 1]
    if controls[0] == 1 and controls[1] == 1:
        target = (target + 1) % 2
    return controls, target


def fanout_classical_values(
    control: IntegerValue, targets: IntegerValue, width: int | Expr
) -> tuple[IntegerValue, IntegerValue | Expr]:
    """Use control truthiness and XOR the original width mask when active.

    Inactive evaluation returns the original target object without evaluating
    the width expression. This differs from CNOT's modulo-two arithmetic and
    Toffoli's exact equality tests, especially for non-bit diagnostic inputs.
    """
    if control:
        return control, (2**width - 1) ^ targets
    return control, targets
