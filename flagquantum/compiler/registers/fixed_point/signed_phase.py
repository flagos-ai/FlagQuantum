# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ComplexPhaseOracle.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mean_estimation/complex_phase_oracle.py

"""Apply a signed binary fraction as a phase without modifying its registers."""

from math import ldexp, pi

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..control.control_predicate import ControlRegister
from ..control.controlled_component import controlled_component_ir


def signed_fractional_phase_ir(
    sign: int, magnitude: tuple[int, ...], *, adjoint: bool = False
) -> tuple[Instruction, ...]:
    """Apply exp(±iπ f) for a sign bit and big-endian fractional word.

    Zero sign gives a positive phase. One sign gives a negative phase. Visit
    the magnitude from most to least significant, using zero-control then
    one-control on each bit. The native phase primitive owns no work or slots.
    """
    if type(adjoint) is not bool:
        raise ValueError("Signed phase direction must be boolean")
    _validate_wires((sign, *magnitude))
    gates: list[Instruction] = []
    for index, target in enumerate(magnitude):
        angle = ldexp(-pi if adjoint else pi, -index - 1)

        def positive(
            control: int, target: int = target, angle: float = angle
        ) -> tuple[Instruction, ...]:
            return (Instruction("cphase", (control, target), {"theta": angle}),)

        gates.extend(
            controlled_component_ir(
                ((sign,),),
                (ControlRegister(1, (0,)),),
                positive,
                (),
                (),
            )
        )
        gates.append(Instruction("cphase", (sign, target), {"theta": -angle}))
    return tuple(gates)
