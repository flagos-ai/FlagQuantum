# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 first_quantization/prepare_t.py.

"""Compose the ordinary kinetic coordinate and two unary momentum calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def kinetic_preparation_program(
    signature: Signature, uniform: ProgramRecord, power: ProgramRecord
) -> ProgramRecord:
    """Keep the original shared power child and its distinct r/s operand roles."""
    builder = ComponentProgramBuilder(signature)
    coordinate = builder.call_program(
        uniform, {"target": builder.inputs["w"]}, entry=uniform.functions[0].name
    )
    unary = builder.import_program(power, entry=power.functions[0].name)
    first = builder.call_dict(unary, {"r": builder.inputs["r"]})
    second = builder.call_dict(unary, {"r": builder.inputs["s"]})
    return builder.finalize(
        {"w": coordinate["target"], "r": first["r"], "s": second["r"]}
    )
