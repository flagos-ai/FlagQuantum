# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, ECAdd.

"""Thread the six source stages while retaining logical bystanders."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def elliptic_curve_addition_program(
    signature: Signature, stages: tuple[ProgramRecord, ...]
) -> ProgramRecord:
    """Compose six exact stages without weakening their one-sided promises."""
    if len(stages) != 6:
        raise ValueError("Elliptic-curve addition requires six source stages")
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    for stage in stages:
        inputs = Signature(stage.functions[0].registers or ()).lefts()
        values = {register.name: state.pop(register.name) for register in inputs}
        state.update(builder.call_program(stage, values, entry=stage.functions[0].name))
    return builder.finalize(state)
