# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 RzViaPhaseGradient.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/rz_via_phase_gradient.py

"""Variable Z-axis rotations through a shared phase-gradient register."""

from ....core.ir import Instruction
from ..arithmetic.addition import measurement_add_or_subtract_ir
from ..layout.register_cast import RegisterCast


def rz_via_phase_gradient_ir(
    target: int,
    angle: tuple[int, ...],
    gradient: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply Rz(4πa/2**n) with an n-bit negative-exponent gradient state.

    The pinned casts preserve the bit patterns. The unsigned angle integer
    a is read from a big-endian word of at most n bits. The arithmetic
    subtracts on zero and adds on one, with the reverse in the adjoint.
    Supply n-1 clean carries and matching classical slots.
    """
    if type(adjoint) is not bool:
        raise ValueError("Rz phase gradient adjoint must be boolean")
    return measurement_add_or_subtract_ir(
        target,
        RegisterCast(len(angle), len(angle)).bind(angle),
        RegisterCast(len(gradient), len(gradient)).bind(gradient),
        carries,
        classical_bits,
        add_when_ctrl_is_on=not adjoint,
    )
