# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call graph follows Qualtran v0.7.0 PRGAViaPhaseGradient.

"""Compose angle-word loading, phase addition and exact XOR erasure."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ..component_builder import ComponentProgramBuilder
from ..lookup.qrom_component import qrom_program
from ..lookup.qrom_table import QromTable
from .phase_gradient_lookup import PhaseGradientLookupCall
from .scaled_phase_component import phase_add_component_program


def phase_gradient_lookup_program(
    table: QromTable, *, adjoint: bool = False
) -> ProgramRecord:
    """Keep source Allocate/QROM/Add/QROM/Free and caller-owned gradient.

    Addition is a source atomic boundary. Existing native integer arithmetic
    executes beneath it, with an exact inverse. QROM uses its own forward
    measured cleanup in either direction, never a reversed measurement stream.
    """
    if type(adjoint) is not bool:
        raise ValueError("Phase lookup direction must be boolean")
    if (
        len(table.shape) != 1
        or len(table.target_bitsizes) != 1
        or table.target_shapes != ((),)
    ):
        raise ValueError("Phase lookup requires one scalar angle dataset")
    width = table.target_bitsizes[0]
    if type(width) is not int or width < 1:
        raise ValueError("Executable phase lookup requires a positive concrete width")
    builder = ComponentProgramBuilder(PhaseGradientLookupCall(table).signature)
    words = dict(builder.inputs)
    words["target0_"] = builder.allocate(width)
    gradient = words.pop("phase_gradient")
    lookup = qrom_program(table)
    words = builder.call_program(lookup, words)
    returned = builder.call_program(
        phase_add_component_program(width, width, sign=-1 if adjoint else 1),
        {"x": words["target0_"], "phase_grad": gradient},
    )
    words["target0_"], gradient = returned["x"], returned["phase_grad"]
    words = builder.call_program(lookup, words)
    words["phase_gradient"] = gradient
    angle = words.pop("target0_")
    assert isinstance(angle, RegisterBinding)
    builder.free(angle)
    return builder.finalize(words)
