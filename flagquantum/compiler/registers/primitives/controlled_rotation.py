# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CRz and CZPowGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Source controlled Z rotations and powers with exact phase conventions."""

from math import pi

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..control.control_predicate import ControlRegister
from ..control.controlled_component import controlled_component_ir
from .and_compute import and_compute_ir, and_uncompute_ir
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


def z_power_with_control_ir(
    target: int,
    exponent: float,
    *,
    control: int | None = None,
    value: int = 1,
    work: int | None = None,
    classical_bit: int | None = None,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Follow ZPow's control dispatch through the measured CZPow decomposition.

    A controlled call needs one clean work qubit and one reusable measurement
    slot. Zero control uses the existing ControlledViaAnd single-bit wrapper.
    Adjoint negates the source exponent before the same control dispatch.
    A bare call uses neither work nor a slot. Measured cleanup restores work
    in each branch; execution belongs to the existing dynamic runtime.
    """
    if type(adjoint) is not bool:
        raise ValueError("Z power direction must be boolean")
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Z power control value must be zero or one")
    signed = _finite_angle(exponent) * (-1 if adjoint else 1)
    if control is None:
        if value != 1 or work is not None or classical_bit is not None:
            raise ValueError("Bare Z power takes no control storage or polarity")
        return pauli_power_ir("z", target, signed)
    if work is None or classical_bit is None:
        raise ValueError("Controlled Z power requires clean work and a result slot")
    _validate_wires((control, target, work))
    return controlled_component_ir(
        ((control,),),
        (ControlRegister(1, (value,)),),
        lambda positive: controlled_z_power_ir(
            positive, target, signed, work, classical_bit
        ),
        (),
        (),
    )
