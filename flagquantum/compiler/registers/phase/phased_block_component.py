# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 block_encoding/phase.py.

"""Compose a source scalar phase and block using explicitly owned exact pairs."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder


def phased_block_component_pair(
    child: ProgramRecord,
    inverse: ProgramRecord,
    scalar: ProgramRecord,
    scalar_inverse: ProgramRecord,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Retain the original zero-port scalar call before the through block call.

    Scalar execution and carrier workspace belong to the supplied existing
    owner, which must restore its private work. Block storage remains owned by
    the caller. Generic inverse uses explicit child and scalar inverse Programs;
    no measurement instructions or scalar primitive sequences are reversed.
    """
    registers = child.functions[0].registers
    if (
        registers is None
        or not Signature(registers).thru_registers_only
        or inverse.functions[0].registers != registers
        or scalar.functions[0].registers != ()
        or scalar_inverse.functions[0].registers != ()
    ):
        raise ValueError(
            "Phased block requires through child and registerless scalar pairs"
        )
    builder = ComponentProgramBuilder(Signature(registers))
    builder.call_program(scalar, {}, entry=scalar.functions[0].name)
    forward = builder.finalize(
        builder.call_program(child, builder.inputs, entry=child.functions[0].name)
    )
    backward = adjoint_component_calls(
        forward, ((child, inverse), (scalar, scalar_inverse))
    )
    return forward, backward
