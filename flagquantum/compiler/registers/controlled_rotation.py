# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CRz and CZPowGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Source controlled Z rotations and powers with exact phase conventions."""

from math import pi

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires
from .global_phase import _finite_angle
from .pauli_rotation import pauli_power_ir


def controlled_rz_ir(
    control: int, target: int, angle: float
) -> tuple[Instruction, ...]:
    """Apply Rz in radians under a positive control with no workspace."""
    _validate_wires((control, target))
    # Keep the source's ZPow exponent conversion before returning to radians.
    half = (_finite_angle(angle) / pi) / 2
    return (
        *pauli_power_ir("z", target, half),
        Instruction("cx", (control, target)),
        *pauli_power_ir("z", target, -half),
        Instruction("cx", (control, target)),
    )


def controlled_z_power_ir(
    control: int, target: int, exponent: float, work: int, classical_bit: int
) -> tuple[Instruction, ...]:
    """Phase the 11 state by pi times exponent and restore a clean work qubit."""
    return (
        *and_compute_ir(control, target, work),
        *pauli_power_ir("z", work, exponent),
        *and_uncompute_ir(control, target, work, classical_bit),
    )
