# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/find_ecc_private_key.py.

"""Compose raw coordinate initialization, two phase calls and dirty releases."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def curve_double_phase_program(
    *,
    initialize_x: ProgramRecord,
    initialize_y: ProgramRecord,
    base_phase: ProgramRecord,
    key_phase: ProgramRecord,
) -> ProgramRecord:
    """Preserve raw source IntState values and explicitly dirty coordinate exits.

    The root owns both coordinate leases. Phase children own their controls and
    private work. Neither source root exposes a classical result. A missing child
    body remains missing rather than becoming a reset or a numerical callback.
    """
    builder = ComponentProgramBuilder(Signature(()))
    x = builder.call_program(initialize_x, {}, entry=initialize_x.functions[0].name)[
        "val"
    ]
    y = builder.call_program(initialize_y, {}, entry=initialize_y.functions[0].name)[
        "val"
    ]
    for phase in (base_phase, key_phase):
        returned = builder.call_program(
            phase, {"x": x, "y": y}, entry=phase.functions[0].name
        )
        x, y = returned["x"], returned["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    builder.free(x, dirty=True)
    builder.free(y, dirty=True)
    return builder.finalize({})
