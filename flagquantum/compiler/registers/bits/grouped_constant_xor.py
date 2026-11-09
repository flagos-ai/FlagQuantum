# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd(XorK).
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py
# XorK: qualtran/bloqs/arithmetic/bitwise.py at the same pinned revision.

"""Structured activation of constant XOR through shared predicate compilation."""

from dataclasses import fields

from ....core.ir import Instruction
from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..arithmetic.arithmetic_declarations import arithmetic_call_graph
from ..control.control_predicate import ControlRegister, bind_control_registers
from ..control.controlled_component import controlled_component_ir
from ..control.controlled_component_resource_model import controlled_component_cost
from .bitwise import _validate_wires, controlled_xor_constant_ir


def grouped_controlled_xor_constant_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    target: tuple[int, ...],
    value: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR an unsigned bit pattern when all stored control words match.

    Target and controls are MSB first. For m control bits, supply max(0,m-1)
    zero-initialized work wires and distinct measurement slots. Every work wire
    returns to zero; the original controls and arbitrary target remain coherent.
    Signed target constants must be encoded into unsigned bit patterns first.
    """
    wires, _ = bind_control_registers(controls, registers)
    _validate_wires((*wires, *target, *workspace))

    def body(predicate: int) -> tuple[Instruction, ...]:
        return controlled_xor_constant_ir(predicate, target, value)

    return controlled_component_ir(controls, registers, body, workspace, classical_bits)


def source_grouped_constant_xor_cost(
    width: int,
    value: int,
    registers: tuple[ControlRegister, ...],
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Query the positive child, then compose original predicate lifetimes.

    These logical QEC categories are separate from the emitted primitive gates.
    Width and constant are concrete Python integers and the constant is an
    unsigned stored pattern; this entry does not accept symbolic execution.
    """
    if type(width) is not int or width < 1 or type(value) is not int:
        raise ValueError(
            "Constant XOR model requires positive width and integer pattern"
        )
    root = ArithmeticCall("controlled_xor_constant", (width,), constant=value)
    child = query_qec_gate_cost(
        root,
        declared_calls=arithmetic_call_graph(root),
        legacy_shims=legacy_shims,
    )
    positive = ComponentCost(
        "ControlledXorK",
        width + 1,
        GateCost(**{field.name: getattr(child, field.name) for field in fields(child)}),
    )
    return controlled_component_cost(
        positive, registers, target_width=width, legacy_shims=legacy_shims
    )
