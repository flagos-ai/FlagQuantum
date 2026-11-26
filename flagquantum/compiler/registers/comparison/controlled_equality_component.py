# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Controlled Equals calls follow locked Qualtran v0.7.0 arithmetic/comparison.py.

"""Compose controlled equality through shared controlled-XOR and predicates."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.multi_control_component import multi_control_x_program


def _controlled_zero_predicate(width: int) -> ProgramRecord:
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("ctrl", bit),
                Register("controls", bit, (width,)),
                Register("target", bit),
            )
        )
    )
    word = builder.inputs["controls"]
    assert isinstance(word, np.ndarray)
    result = builder.call_program(
        multi_control_x_program((1, *((0,) * width))),
        {
            "controls": np.array([builder.inputs["ctrl"], *word]),
            "target": builder.inputs["target"],
        },
    )
    controls = result["controls"]
    assert isinstance(controls, np.ndarray)
    return builder.finalize(
        {"ctrl": controls[0], "controls": controls[1:], "target": result["target"]}
    )


def controlled_equality_program(
    dtype: RegisterType, *, control_value: int = 1
) -> ProgramRecord:
    """Retain both words and XOR equality into a dirty target under one control.

    The source positive control compares the conditionally XORed second word
    against zero, then restores it. Negative activation wraps that body in X.
    Predicate work is private and measurement-erased; words are MSB-first.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled equality requires a binary activation")
    signature = Signature.build_from_dtypes(
        ctrl=quantum_bit_type(), x=dtype, y=dtype, target=quantum_bit_type()
    )
    if control_value == 0:
        builder = ComponentProgramBuilder(signature)
        flip = bit_gate_function(
            "x",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("x", (0,)),)),
        )
        builder.define(flip)
        values = dict(builder.inputs)
        values["ctrl"] = builder.call_dict("x", {"q": values["ctrl"]})["q"]
        values = builder.call_program(controlled_equality_program(dtype), values)
        values["ctrl"] = builder.call_dict("x", {"q": values["ctrl"]})["q"]
        return builder.finalize(values)
    builder = ComponentProgramBuilder(signature)
    xor = controlled_bitwise_word_program(dtype, "xor")
    values = builder.call_program(
        xor, {name: builder.inputs[name] for name in ("ctrl", "x", "y")}
    )
    y = values["y"]
    assert isinstance(y, RegisterBinding)
    bits = builder.split(y)
    compared = builder.call_program(
        _controlled_zero_predicate(len(bits)),
        {"ctrl": values["ctrl"], "controls": bits, "target": builder.inputs["target"]},
    )
    controls: BindingCollection = compared["controls"]
    assert isinstance(controls, np.ndarray)
    restored = builder.call_program(
        xor,
        {
            "ctrl": compared["ctrl"],
            "x": values["x"],
            "y": builder.join(controls, dtype),
        },
    )
    return builder.finalize({**restored, "target": compared["target"]})
