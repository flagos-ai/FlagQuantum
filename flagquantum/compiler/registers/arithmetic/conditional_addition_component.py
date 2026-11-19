# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledAddOrSubtract.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py

"""Compose conditional sum/difference through shared typed arithmetic calls."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .addition_component import addition_component_program


def conditional_addition_program(
    a_dtype: RegisterType,
    b_dtype: RegisterType | None = None,
    *,
    add_when_control_is_on: bool = True,
) -> ProgramRecord:
    """Retain ctrl/a and conditionally add or subtract a into b, MSB first.

    The default b descriptor is a's descriptor. For True, control one adds and
    control zero subtracts. False exchanges these sectors and gives the exact
    adjoint through the original forward Add. Carries belong to the shared Add
    graph and are measured, reset and released. A narrower signed a retains the
    original circuit's zero-extension behavior. All data words remain live.
    """
    if type(add_when_control_is_on) is not bool:
        raise ValueError("Conditional addition polarity requires a boolean")
    target_dtype = a_dtype if b_dtype is None else b_dtype
    addition = addition_component_program(
        executable_register_width(a_dtype),
        executable_register_width(target_dtype),
        name="conditional_add",
        left_dtype=a_dtype,
        right_dtype=target_dtype,
    )
    complement = controlled_bitwise_word_program(target_dtype, "bitwise_not")
    bit = quantum_bit_type()
    flip = bit_gate_function(
        "conditional_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=bit, a=a_dtype, b=target_dtype), (flip,)
    )
    control, a, b = (builder.inputs[name] for name in ("ctrl", "a", "b"))
    if add_when_control_is_on:
        control = builder.call_dict(flip.name, {"q": control})["q"]
    outputs = builder.call_program(complement, {"ctrl": control, "x": b})
    control, b = outputs["ctrl"], outputs["x"]
    outputs = builder.call_program(addition, {"a": a, "b": b}, entry="conditional_add")
    a, b = outputs["a"], outputs["b"]
    outputs = builder.call_program(complement, {"ctrl": control, "x": b})
    control, b = outputs["ctrl"], outputs["x"]
    if add_when_control_is_on:
        control = builder.call_dict(flip.name, {"q": control})["q"]
    return builder.finalize({"ctrl": control, "a": a, "b": b})
