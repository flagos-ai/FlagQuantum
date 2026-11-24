# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call construction follows Qualtran v0.7.0 reflections/prepare_identity.py.

"""Own typed identity selection calls independently of algorithm declarations."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..primitives.identity import register_identity_program


def identity_preparation_program(registers: tuple[Register, ...]) -> ProgramRecord:
    """Retain one source Identity call per concrete scalar through word.

    Words are MSB-first. No work or measurements are introduced. The original
    source does not supply executable shaped, one-sided or zero-width bodies.
    """
    if any(reg.shape_symbolic or reg.side != RegisterSide.THRU for reg in registers):
        raise ValueError("Identity preparation requires scalar through ports")
    if any(reg.bitsize == 0 for reg in registers):
        raise ValueError("Identity preparation source has no zero-width body")
    if any(reg.dtype.num_cbits != 0 for reg in registers):
        raise ValueError("Identity preparation requires quantum selection ports")
    builder = ComponentProgramBuilder(Signature(registers))
    result = dict(builder.inputs)
    for reg in registers:
        if type(reg.bitsize) is not int:
            raise ValueError("Stored identity preparation requires concrete widths")
        result[reg.name] = builder.call_program(
            register_identity_program(reg.bitsize), {"q": result[reg.name]}
        )["q"]
    return builder.finalize(result)
