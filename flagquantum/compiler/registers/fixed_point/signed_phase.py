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
    sign: int,
    magnitude: tuple[int, ...],
    *,
    adjoint: bool = False,
    control: int | None = None,
    control_value: int = 1,
    work: int | None = None,
    classical_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Apply exp(±iπ f) for a sign bit and big-endian fractional word.

    Zero sign gives a positive phase. One sign gives a negative phase. Visit
    the magnitude from most to least significant, using zero-control then
    one-control on each bit. An outer control uses one reusable clean wire
    and measurement slot to combine the two conditions.
    """
    if type(adjoint) is not bool:
        raise ValueError("Signed phase direction must be boolean")
    outer: tuple[tuple[int, ...], ...]
    pattern: tuple[ControlRegister, ...]
    scratch: tuple[int, ...]
    slots: tuple[int, ...]
    if control is None:
        if work is not None or classical_bit is not None or control_value != 1:
            raise ValueError("Uncontrolled signed phase takes no predicate storage")
        _validate_wires((sign, *magnitude))
        outer, pattern, scratch, slots = (), (), (), ()
    else:
        if work is None or classical_bit is None:
            raise ValueError(
                "Controlled signed phase requires one predicate wire and slot"
            )
        _validate_wires((control, sign, *magnitude, work))
        outer = ((control,),)
        pattern = (ControlRegister(1, (control_value,)),)
        scratch, slots = (work,), (classical_bit,)

    gates: list[Instruction] = []
    for index, target in enumerate(magnitude):
        angle = ldexp(-pi if adjoint else pi, -index - 1)
        for sign_value, theta in ((0, angle), (1, -angle)):

            def phase(
                predicate: int, target: int = target, theta: float = theta
            ) -> tuple[Instruction, ...]:
                return (Instruction("cphase", (predicate, target), {"theta": theta}),)

            gates.extend(
                controlled_component_ir(
                    (*outer, (sign,)),
                    (*pattern, ControlRegister(1, (sign_value,))),
                    phase,
                    scratch,
                    slots,
                )
            )
    return tuple(gates)
