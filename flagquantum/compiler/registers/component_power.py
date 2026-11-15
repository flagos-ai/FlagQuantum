# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Power.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py

"""Repeat an owned register component through the existing call builder."""

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.operator_schema import MAX_POWER_REPEATS
from .component_builder import ComponentProgramBuilder


def power_word_program(
    child: FunctionRecord,
    repetitions: int,
    *,
    definitions: tuple[FunctionRecord, ...] = (),
) -> ProgramRecord:
    """Thread all named in-place child outputs into its next call.

    The child and its definitions carry the actual register and workspace
    contracts. Use the resource graph owner for symbolic multiplicity.
    """
    if child.registers is None or any(
        register.side != RegisterSide.THRU for register in child.registers
    ):
        raise ValueError("Power requires a typed child with only THRU registers")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Power requires a positive integer repetition count")
    if repetitions > MAX_POWER_REPEATS:
        raise ValueError(f"Power exceeds {MAX_POWER_REPEATS} repeated calls")

    builder = ComponentProgramBuilder(Signature(child.registers), (child, *definitions))
    bindings = builder.inputs
    for _ in range(repetitions):
        bindings = builder.call_dict(child.name, bindings)
    return builder.finalize(bindings)
