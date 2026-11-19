# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledAddOrSubtract.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py

"""Compose conditional sum/difference through shared typed arithmetic calls."""

from dataclasses import replace
from functools import partial

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder, control_component_hierarchy
from ..component_gate_function import bit_gate_function
from .addition_component import addition_component_program


def _preserve_control(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    callee: str,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    return control, tuple(builder.call_dict(callee, values).values())


def _controlled_x(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict(
        "conditional_cx", {"ctrl": control, "target": values["q"]}
    )
    return outputs["ctrl"], (outputs["target"],)


def _controlled_complement(dtype: RegisterType) -> ProgramRecord:
    record = bitwise_word_program(dtype, "bitwise_not")
    bit = quantum_bit_type()
    cx = bit_gate_function(
        "conditional_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    adders: dict[str, ControlAdder] = {}
    for function in record.functions:
        assert function.registers is not None
        if function.name.endswith("bitwise_x"):
            adders[function.name] = _controlled_x
        elif any(register.side != RegisterSide.THRU for register in function.registers):
            adders[function.name] = partial(_preserve_control, callee=function.name)
    return control_component_hierarchy(
        replace(record, functions=(*record.functions, cx)), adders
    )


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
    complement = _controlled_complement(target_dtype)
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
