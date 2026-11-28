# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 first_quantization/prepare_uv.py.

"""Compose potential preparation with separately owned momentum and nuclear words."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def potential_preparation_program(
    signature: Signature,
    momentum: ProgramRecord,
    nuclear: ProgramRecord,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Preserve the source nu-then-zeta calls and nuclear-index bystander."""
    builder = ComponentProgramBuilder(signature)
    if adjoint:
        nuclear_result = builder.call_program(
            nuclear, {"l": builder.inputs["l"]}, entry=nuclear.functions[0].name
        )
    momentum_result = builder.call_program(
        momentum,
        {name: builder.inputs[name] for name in ("mu", "nu", "m", "flag_nu")},
        entry=momentum.functions[0].name,
    )
    if not adjoint:
        nuclear_result = builder.call_program(
            nuclear, {"l": builder.inputs["l"]}, entry=nuclear.functions[0].name
        )
    return builder.finalize({**momentum_result, **nuclear_result})
