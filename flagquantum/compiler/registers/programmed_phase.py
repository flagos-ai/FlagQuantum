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


def programmed_phase_rotation_ir(
    data: int,
    work: int,
    outcomes: tuple[int, ...],
    *,
    exponent: float,
    epsilon: float = 1e-11,
    apply_final_correction: bool = False,
) -> tuple[Instruction, ...]:
    """Emit a bounded programmed rotation with one reusable clean work wire.

    Each outcome slot belongs to one round. Zero indicates success or padding.
    With correction disabled, a one in every slot indicates failure. The work
    wire returns to zero. The flat IR measures the idle zero wire in skipped
    rounds, since measurement itself cannot be predicated.
    """
    _validate_wires((data, work))
    _validate_wires(outcomes)
    if type(apply_final_correction) is not bool:
        raise ValueError("programmed phase correction flag must be boolean")
    if type(epsilon) not in (int, float) or not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("programmed phase requires a finite positive epsilon")

    tolerance = epsilon / (len(outcomes) + apply_final_correction)
    instructions: list[Instruction] = []
    failures: tuple[tuple[int, int], ...] = ()
    for bit in outcomes:
        metadata = {"conditions": failures} if failures else {}
        prepare = programmed_phase_ir(work, exponent=exponent, epsilon=tolerance)
        instructions.extend(
            Instruction(gate.name, gate.wires, gate.params, metadata=metadata)
            for gate in prepare
        )
        instructions.append(Instruction("cx", (data, work), metadata=metadata))
        instructions.append(
            Instruction(
                "measure", (work,), metadata={"is_dynamic": True, "classical_bit": bit}
            )
        )
        failures = tuple(sorted((*failures, (bit, 1))))
        instructions.append(
            Instruction("x", (work,), metadata={"conditions": failures})
        )
        exponent *= 2
    if apply_final_correction:
        angle = pi * exponent
        if not isfinite(angle):
            raise ValueError("programmed phase correction angle must be finite")
        instructions.append(
            Instruction(
                "phase",
                (data,),
                params={"theta": angle},
                metadata={"conditions": failures},
            )
        )
    return tuple(instructions)
