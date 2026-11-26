# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 PrepareFirstQuantizationWithProj.

"""Compose projectile PREPARE while retaining its original plus-state bystander."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def projectile_preparation_program(
    signature: Signature,
    stages: tuple[ProgramRecord, ProgramRecord, ProgramRecord, ProgramRecord],
) -> ProgramRecord:
    """Thread term, particle, kinetic and potential stages in source order."""
    builder = ComponentProgramBuilder(signature)
    entries = tuple(
        builder.import_program(stage, entry=stage.functions[0].name) for stage in stages
    )
    state = dict(builder.inputs)
    for entry, names in zip(
        entries[:3],
        (("tuv", "tepm", "uv", "flags"), ("i", "j"), ("w", "w_mean", "r", "s")),
        strict=True,
    ):
        state.update(builder.call_dict(entry, {name: state[name] for name in names}))
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
