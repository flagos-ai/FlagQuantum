# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CRz and CZPowGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Source controlled Z rotations and powers with exact phase conventions."""

from math import isfinite, pi

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires


def _finite_angle(value: float, scale: float = 1.0) -> float:
    if type(value) in (int, float):
        try:
            angle = scale * value
            if isfinite(angle):
                return angle
        except OverflowError:
            pass
    raise ValueError("Controlled rotation requires a finite phase angle")


def controlled_rz_ir(
    control: int, target: int, angle: float
) -> tuple[Instruction, ...]:
    """Apply Rz in radians under a positive control with no workspace."""
    _validate_wires((control, target))
    # Keep the source's ZPow exponent conversion before returning to radians.
    half = pi * ((_finite_angle(angle) / pi) / 2)
    return (
        Instruction("phase", (target,), params={"theta": half}),
        Instruction("cx", (control, target)),
        Instruction("phase", (target,), params={"theta": -half}),
        Instruction("cx", (control, target)),
    )


def controlled_z_power_ir(
    control: int, target: int, exponent: float, work: int, classical_bit: int
) -> tuple[Instruction, ...]:
    """Phase the 11 state by pi times exponent and restore a clean work qubit."""
    angle = _finite_angle(exponent, pi)
    return (
        *and_compute_ir(control, target, work),
        Instruction("phase", (work,), params={"theta": angle}),
        *and_uncompute_ir(control, target, work, classical_bit),
    )
