# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ZPowConstViaPhaseGradient.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/zpow_via_phase_gradient.py

"""Constant Z rotations through a shared phase-gradient register."""

from math import isfinite

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires, controlled_xor_constant_ir
from .phase_addition import phase_add_ir


def zpow_phase_gradient_constant(exponent: float, width: int) -> int:
    """Encode the original unsigned, truncated and wrapping exponent/2."""
    if type(width) is not int or width < 1:
        raise ValueError("Gradient width must be a positive integer")
    if type(exponent) not in (int, float) or not isfinite(exponent) or exponent < 0:
        raise ValueError("Z power requires a finite nonnegative exponent")
    numerator, denominator = (exponent / 2).as_integer_ratio()
    return ((numerator << width) // denominator) % (1 << width)


def zpow_via_phase_gradient_ir(
    target: int,
    gradient: tuple[int, ...],
    angle_work: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    exponent: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Kick back a constant Z power using a negative-exponent gradient state.

    Use an n-bit clean angle word and n-1 clean carries with matching slots.
    The original load and unload are retained, with an exact measured adder
    (or its adjoint) between them. The gradient itself is caller-owned.
    """
    n = len(gradient)
    value = zpow_phase_gradient_constant(exponent, n)
    if type(adjoint) is not bool:
        raise ValueError("Z power adjoint must be boolean")
    if len(angle_work) != n:
        raise ValueError("Z power requires one clean angle wire per gradient bit")
    _validate_wires((target, *gradient, *angle_work, *carries))
    load = controlled_xor_constant_ir(target, angle_work, value)
    addition = phase_add_ir(
        angle_work, gradient, carries, classical_bits, sign=-1 if adjoint else 1
    )
    return (*load, *addition, *load)
