# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Scalar meaning follows locked Qualtran v0.7.0 basic_gates/global_phase.py.

"""Own the existing exact scalar carrier as a registerless typed Program."""

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ....core.parameters import Parameter, ParameterExpression
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .global_phase import scalar_phase_ir


def scalar_phase_program(
    angle: float | Parameter | ParameterExpression,
) -> ProgramRecord:
    """Allocate, phase, restore and free one private carrier when needed.

    The entry has no logical ports. A zero constant phase allocates no work.
    Physical gates are the accepted scalar_phase_ir recipe, independently of
    the source's zero-cost logical model. No Program opcode is introduced.
    """
    gates = scalar_phase_ir(0, angle)
    builder = ComponentProgramBuilder(Signature(()))
    if gates:
        carrier = builder.allocate(1)
        phase = bit_gate_function(
            "scalar_phase", tuple(Signature.build(q=1)), CircuitIR(1, gates)
        )
        restored = builder.call_program(
            ProgramRecord((phase,)), {"q": carrier}, entry=phase.name
        )["q"]
        assert isinstance(restored, RegisterBinding)
        builder.free(restored)
    return builder.finalize({})
