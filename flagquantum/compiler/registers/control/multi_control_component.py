# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiControlPauliBase.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_control_pauli.py

"""Compose literal multi-control X through the existing predicate owner."""

from dataclasses import replace

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import control_component_via_predicate
from ..component_gate_function import bit_gate_function
from .control_predicate import ControlRegister


def _add_x(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    result = builder.call_dict("controlled_x", {"ctrl": control, "target": values["q"]})
    return result["ctrl"], (result["target"],)


def multi_control_x_program(values: tuple[int, ...]) -> ProgramRecord:
    """XOR a literal activation into a dirty target, preserving ordered controls.

    Controls are a row-major QBit array. The shared predicate owns and erases
    its clean work. The empty activation applies X without a controls port.
    Invalid nonbinary activations raise ValueError.
    """
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError("Multi-control X requires binary activation values")
    bit = quantum_bit_type()
    x = bit_gate_function(
        "x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    if values:
        cx = bit_gate_function(
            "controlled_x",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        )
        body = ComponentProgramBuilder(Signature.build_from_dtypes(q=bit), (x, cx))
        target = body.call_dict("x", body.inputs)["q"]
        controlled = control_component_via_predicate(
            body.finalize({"q": target}),
            {"main": _add_x},
            (Register("ctrl", bit, (len(values),)),),
            (ControlRegister(1, values),),
        )
        controlled = replace(
            controlled,
            functions=tuple(f for f in controlled.functions if f.name != "x"),
        )
        builder = ComponentProgramBuilder(
            Signature(
                (Register("controls", bit, (len(values),)), Register("target", bit))
            )
        )
        result = builder.call_program(
            controlled,
            {"ctrl": builder.inputs["controls"], "q": builder.inputs["target"]},
        )
        return builder.finalize({"controls": result["ctrl"], "target": result["q"]})
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(target=bit), (x,))
    return builder.finalize(
        {"target": builder.call_dict("x", {"q": builder.inputs["target"]})["q"]}
    )
