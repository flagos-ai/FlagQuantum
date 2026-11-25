# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ArcTan.apply and QFxp.to_fixed_width_int.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arctan.py

"""Compose the original scalar XOR callback with the optional real codec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..simulation.classical_arctangent import normalized_arctangent_value
from .fixed_point import fixed_point_real_to_integer

if TYPE_CHECKING:
    from sympy import Expr


def source_arctangent_classical_values(
    selection: int, sign: int, target: int, target_width: int | Expr
) -> tuple[int, int, int]:
    """XOR the source sign and magnitude, retaining its fixed-point wrapping.

    Selection is returned unchanged. Output integers are not masked, and the
    original binary64 saturation and optional codec failures are preserved.
    This scalar callback has no quantum lookup-width restriction.
    """
    output = normalized_arctangent_value(selection)
    encoded = fixed_point_real_to_integer(abs(output), target_width, target_width)
    return selection, sign ^ int(output < 0), target ^ encoded
