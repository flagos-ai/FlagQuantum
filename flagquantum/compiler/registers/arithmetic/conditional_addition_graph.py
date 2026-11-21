# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/controlled_add_or_subtract.py.

"""Own conditional complement/Add/complement wiring around supplied definitions."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function


def conditional_addition_graph(
    addition: ProgramRecord,
    complement: ProgramRecord,
    *,
    add_when_control_is_on: bool,
) -> ProgramRecord:
    """Retain source control flips and three typed arithmetic calls.

    Addition has a/b THRU words and owns its carry workspace. Complement has
    ctrl/x THRU ports. This wiring owns no workspace and retains both logical
    words. Concrete and symbolic consumers supply their existing definitions.
    """
    registers = addition.functions[0].registers
    assert registers is not None
    bit = quantum_bit_type()
    flip = bit_gate_function(
        "conditional_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            ctrl=bit, a=registers[0].dtype, b=registers[1].dtype
        ),
        (flip,),
    )
    control, a, b = (builder.inputs[name] for name in ("ctrl", "a", "b"))
    if add_when_control_is_on:
        control = builder.call_dict(flip.name, {"q": control})["q"]
    outputs = builder.call_program(complement, {"ctrl": control, "x": b})
    control, b = outputs["ctrl"], outputs["x"]
    outputs = builder.call_program(
        addition, {"a": a, "b": b}, entry=addition.functions[0].name
    )
    a, b = outputs["a"], outputs["b"]
    outputs = builder.call_program(complement, {"ctrl": control, "x": b})
    control, b = outputs["ctrl"], outputs["x"]
    if add_when_control_is_on:
        control = builder.call_dict(flip.name, {"q": control})["q"]
    return builder.finalize({"ctrl": control, "a": a, "b": b})
