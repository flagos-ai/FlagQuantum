# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 black_box_prepare.py.

"""Keep the packed PREPARE boundary around domain-owned stored dependencies."""

from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.layout.packed_oracle_component import (
    packed_oracle_component_program,
)
from ...core.ir.program import ProgramRecord
from .black_box_prepare import PackedPrepareDeclaration


def packed_prepare_program(
    declaration: PackedPrepareDeclaration,
    prepare: ProgramRecord,
    *,
    inverse: ProgramRecord | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain the existing entry through the shared typed packing owner."""
    if type(adjoint) is not bool:
        raise ValueError("Packed preparation direction must be boolean")
    if declaration.selection_width == 0:
        builder = ComponentProgramBuilder(declaration.signature)
        return builder.finalize(builder.inputs)
    if prepare.functions[0].registers != tuple(declaration.prepare.signature):
        raise ValueError("Packed preparation body must match the declared signature")
    groups = tuple(
        (name, registers)
        for name, registers in (
            ("selection", declaration.prepare.selection_registers),
            ("junk", declaration.prepare.junk_registers),
        )
        if any(reg.total_qubits() != 0 for reg in registers)
    )
    return packed_oracle_component_program(
        declaration.signature, groups, prepare, inverse=inverse, adjoint=adjoint
    )
