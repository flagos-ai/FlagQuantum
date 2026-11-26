# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 ordinary/projectile PREPARE.

"""Compose both first-quantization PREPARE roots while retaining its original plus-state bystander."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def first_quantization_preparation_program(
    signature: Signature,
    stages: tuple[ProgramRecord, ProgramRecord, ProgramRecord, ProgramRecord],
) -> ProgramRecord:
    """Thread term, particle, kinetic and potential stages in source order."""
    builder = ComponentProgramBuilder(signature)
    entries = tuple(
        builder.import_program(stage, entry=stage.functions[0].name) for stage in stages
    )
    state = dict(builder.inputs)
    for entry, stage in zip(entries[:3], stages[:3], strict=True):
        ports = Signature(stage.functions[0].registers or ()).lefts()
        state.update(
            builder.call_dict(entry, {port.name: state[port.name] for port in ports})
        )
    result = builder.call_dict(
        entries[3],
        {
            "mu": state["mu"],
            "nu": np.array(
                [state[name] for name in ("nu_x", "nu_y", "nu_z")], dtype=object
            ),
            "m": state["m"],
            "l": state["l"],
            "flag_nu": state["succ_nu"],
        },
    )
    state.update({name: result[name] for name in ("mu", "m", "l")})
    state["succ_nu"] = result["flag_nu"]
    for index, name in enumerate(("nu_x", "nu_y", "nu_z")):
        state[name] = np.asarray(result["nu"])[index]
    return builder.finalize(state)
