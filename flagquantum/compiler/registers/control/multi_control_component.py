# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiControlPauliBase.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_control_pauli.py

"""Compose literal multi-control X and Z through the existing predicate owner."""

from dataclasses import replace
from typing import Literal

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import control_component_via_predicate
from ..component_gate_function import bit_gate_function
from .control_predicate import ControlRegister


def positive_controlled_multi_z_program(values: tuple[int, ...]) -> ProgramRecord:
    """Preserve Controlled(MultiControlZ)'s merged scalar/array control call.

    The source generic root calls one ControlledViaAnd(Z) with two clauses;
    it does not replace that call with a flattened control-array wrapper.
    """
    if not values or any(
        type(value) is not int or value not in (0, 1) for value in values
    ):
        raise ValueError(
            "Controlled MultiControlZ requires nonempty binary activations"
        )
    bit = quantum_bit_type()
    z = bit_gate_function(
        "z", (Register("q", bit),), CircuitIR(1, (Instruction("z", (0,)),))
    )
    cz = bit_gate_function(
        "positive_z",
        tuple(Signature.build(q1=1, q2=1)),
        CircuitIR(2, (Instruction("cz", (0, 1)),)),
    )
    body = ComponentProgramBuilder(Signature.build(q=1), (z, cz))
    target = body.call_dict(z.name, body.inputs)["q"]

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        arguments: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_dict(cz.name, {"q1": control, "q2": arguments["q"]})
        return outputs["q1"], (outputs["q2"],)

    combined = control_component_via_predicate(
        body.finalize({"q": target}),
        {"main": add},
        (Register("ctrl1", bit), Register("ctrl2", bit, (len(values),))),
        (ControlRegister(1, (1,)), ControlRegister(1, values)),
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("ctrl", bit),
                Register("controls", bit, (len(values),)),
                Register("target", bit),
            )
        )
    )
    outputs = builder.call_program(
        combined,
        {
            "ctrl1": builder.inputs["ctrl"],
            "ctrl2": builder.inputs["controls"],
            "q": builder.inputs["target"],
        },
    )
    return builder.finalize(
        {"ctrl": outputs["ctrl1"], "controls": outputs["ctrl2"], "target": outputs["q"]}
    )


def multi_control_x_program(values: tuple[int, ...]) -> ProgramRecord:
    """XOR a binary activation into a dirty target, retaining ordered controls."""
    return _multi_control_program(values, "x")


def multi_control_z_program(values: tuple[int, ...]) -> ProgramRecord:
    """Phase a target under ordered binary controls using the shared AND erasure.

    Controls are a row-major QBit array. The predicate owns its clean work.
    Empty controls apply Z without a controls port. Invalid activations fail.
    """
    return _multi_control_program(values, "z")


def controlled_word_x_program(values: tuple[int, ...]) -> ProgramRecord:
    """Preserve the source opaque control-word port and scalar q target.

    The same predicate owner handles word and array layouts. Packing adds no
    gates; the returned root is the actual word predicate/positive-X/erase chain.
    """
    if not values:
        raise ValueError("Word-controlled X requires a nonempty activation")
    return _multi_control_program(values, "x", word_control=True)


def _multi_control_program(
    values: tuple[int, ...], axis: Literal["x", "z"], *, word_control: bool = False
) -> ProgramRecord:
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(
            f"Multi-control {axis.upper()} requires binary activation values"
        )
    bit = quantum_bit_type()
    target_gate = bit_gate_function(
        axis, (Register("q", bit),), CircuitIR(1, (Instruction(axis, (0,)),))
    )
    if values:
        control_name, target_name = ("ctrl", "target") if axis == "x" else ("q1", "q2")
        controlled_gate = bit_gate_function(
            f"controlled_{axis}",
            (Register(control_name, bit), Register(target_name, bit)),
            CircuitIR(2, (Instruction(f"c{axis}", (0, 1)),)),
        )
        body = ComponentProgramBuilder(
            Signature.build_from_dtypes(q=bit), (target_gate, controlled_gate)
        )
        target = body.call_dict(axis, body.inputs)["q"]

        def add_controlled(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            result = builder.call_dict(
                f"controlled_{axis}", {control_name: control, target_name: values["q"]}
            )
            return result[control_name], (result[target_name],)

        control_port = (
            Register("ctrl", opaque_register_type(len(values)))
            if word_control
            else Register("ctrl", bit, (len(values),))
        )
        pattern = (
            ControlRegister(
                len(values),
                (
                    sum(
                        value << (len(values) - i - 1) for i, value in enumerate(values)
                    ),
                ),
            )
            if word_control
            else ControlRegister(1, values)
        )
        controlled = control_component_via_predicate(
            body.finalize({"q": target}),
            {"main": add_controlled},
            (control_port,),
            (pattern,),
        )
        controlled = replace(
            controlled,
            functions=tuple(f for f in controlled.functions if f.name != axis),
        )
        if word_control:
            return controlled
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
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(target=bit), (target_gate,)
    )
    return builder.finalize(
        {"target": builder.call_dict(axis, {"q": builder.inputs["target"]})["q"]}
    )
