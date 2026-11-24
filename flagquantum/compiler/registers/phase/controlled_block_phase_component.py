# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 block_encoding/phase.py and
# basic_gates/global_phase.py:get_ctrl_system.

"""Compose a positive-controlled phase with an explicitly owned controlled block."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..primitives.pauli_rotation_component import RotationAngle, z_power_function


def controlled_block_phase_program(
    child: ProgramRecord,
    exponent: RotationAngle,
    *,
    inverse: ProgramRecord | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Follow the source controlled visitor: controlled child, then ZPow.

    The child owns positive-control specialization and any private workspace.
    All outer ports remain caller-owned. An inverse requires the exact child
    inverse; it reverses calls rather than reversing measurement instructions.
    No registerless phase declaration is constructed or admitted here.
    Although Phase inserts the disconnected scalar first, the original generic
    control visitor visits the block before that scalar and threads its control
    successor into ZPow. Preserve that executed dependency order.
    """
    registers = child.functions[0].registers
    if (
        type(adjoint) is not bool
        or not registers
        or not Signature(registers).thru_registers_only
        or registers[0].shape
        or registers[0].dtype != quantum_bit_type()
    ):
        raise ValueError("Controlled phase requires a leading through control bit")
    if adjoint and inverse is None:
        raise ValueError("Controlled phase requires an explicitly owned child inverse")
    phase = ProgramRecord((z_power_function("phase", exponent),))
    builder = ComponentProgramBuilder(Signature(registers))
    state = builder.call_program(
        child, dict(builder.inputs), entry=child.functions[0].name
    )
    control_name = registers[0].name
    state[control_name] = builder.call_program(
        phase, {"q": state[control_name]}, entry="phase"
    )["q"]
    record = builder.finalize(state)
    if not adjoint:
        return record
    assert inverse is not None
    phase_inverse = ProgramRecord((z_power_function("phase_inverse", -exponent),))
    return adjoint_component_calls(record, ((child, inverse), (phase, phase_inverse)))
