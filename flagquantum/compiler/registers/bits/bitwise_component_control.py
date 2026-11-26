# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled and bitwise XorK/BitwiseNot.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Share literal bitwise control propagation across arithmetic consumers."""

from dataclasses import replace
from functools import partial
from typing import Literal

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder, control_component_hierarchy
from ..component_gate_function import bit_gate_function
from .bitwise_component import bitwise_word_program


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
        "bitwise_controlled_x", {"ctrl": control, "target": values["q"]}
    )
    return outputs["ctrl"], (outputs["target"],)


def _controlled_cx(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    controls, target = builder.call_tuple(
        "bitwise_toffoli",
        {"ctrl": np.array([control, values["ctrl"]]), "target": values["target"]},
    )
    assert isinstance(controls, np.ndarray)
    return controls[0], (controls[1], target)


def controlled_bitwise_word_program(
    dtype: RegisterType,
    operation: Literal["bitwise_not", "xor_constant", "xor"],
    *,
    pattern: int | None = None,
) -> ProgramRecord:
    """Control complement, encoded XOR or two-word XOR through actual child calls."""
    record = bitwise_word_program(dtype, operation, pattern=pattern)
    bit = quantum_bit_type()
    cx = bit_gate_function(
        "bitwise_controlled_x",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    if operation == "xor":
        cx = bit_gate_function(
            "bitwise_toffoli",
            (Register("ctrl", bit, (2,)), Register("target", bit)),
            CircuitIR(3, (Instruction("ccx", (0, 1, 2)),)),
        )
    adders: dict[str, ControlAdder] = {}
    for function in record.functions:
        assert function.registers is not None
        if function.name.endswith("bitwise_x"):
            adders[function.name] = _controlled_x
        elif function.name.endswith("bitwise_cx"):
            adders[function.name] = _controlled_cx
        elif any(register.side != RegisterSide.THRU for register in function.registers):
            adders[function.name] = partial(_preserve_control, callee=function.name)
    return control_component_hierarchy(
        replace(record, functions=(*record.functions, cx)), adders
    )
