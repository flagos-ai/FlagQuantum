# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ZPowProgrammedAncilla.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/programmable_ancilla_rotation.py

"""Prepare a reusable phase resource on a clean output wire."""

from math import isfinite, pi

from ...core.ir import Instruction
from .bitwise import _validate_wires


def programmed_phase_ir(
    output: int, *, exponent: float, epsilon: float = 1e-11
) -> tuple[Instruction, ...]:
    """Bind source Allocate/H/ZPow to an already allocated clean output.

    The output carries the prepared state. Epsilon is the source synthesis
    tolerance. Native IR represents the exact phase before synthesis.
    """
    _validate_wires((output,))
    if type(exponent) not in (int, float) or not isfinite(exponent):
        raise ValueError("programmed phase requires a finite exponent")
    if type(epsilon) not in (int, float) or not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("programmed phase requires a finite positive epsilon")
    angle = pi * exponent
    if not isfinite(angle):
        raise ValueError("programmed phase angle must be finite")
    return Instruction("h", (output,)), Instruction(
        "phase", (output,), params={"theta": angle}
    )
