# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled(Xor) and ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py
# Control wrappers: qualtran/_infra/controlled.py and
# qualtran/bloqs/mcmt/controlled_via_and.py at the same revision.

"""Controlled register XOR with shared activation and original atomic Toffolis."""

from dataclasses import fields

from ....core.ir import Instruction
from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..arithmetic.arithmetic_declarations import arithmetic_call_graph
from ..control.control_predicate import ControlRegister, bind_control_registers
from ..control.controlled_component import controlled_component_ir
from ..control.controlled_component_resource_model import controlled_component_cost
from ..primitives.controlled_not import cnot_with_controls_ir
from .bitwise import _validate_wires


def controlled_xor_ir(
    control: int, source: tuple[int, ...], target: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """XOR source into target under one positive control, preserving bit order.

    Equal nonempty registers and the control must be globally distinct. Toffoli
    stays atomic, matching the original source's declared compilation target.
    There are no work wires or measurements for this positive-controlled body.
    """
    if not source or len(source) != len(target):
        raise ValueError("Controlled XOR requires equal nonempty registers")
    _validate_wires((control, *source, *target))
    return tuple(
        gate
        for left, right in zip(source, target, strict=True)
        for gate in cnot_with_controls_ir(left, right, controls=(control,), values=(1,))
    )


def grouped_controlled_xor_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    source: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Bind control words to one predicate around the whole positive XOR body.

    All words are MSB first. For m controls, max(0,m-1) clean work wires and
    measurement slots belong to the caller. Controls and source are preserved;
    the arbitrary target changes and all work returns to zero. XOR is its own
    inverse. Classical slots retain their final measured values.
    """
    wires, _ = bind_control_registers(controls, registers)
    _validate_wires((*wires, *source, *target, *workspace))

    def body(predicate: int) -> tuple[Instruction, ...]:
        return controlled_xor_ir(predicate, source, target)

    return controlled_component_ir(controls, registers, body, workspace, classical_bits)


def source_grouped_xor_cost(
    width: int,
    registers: tuple[ControlRegister, ...],
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Compose the source width-many Toffolis and shared predicate lifetimes.

    Both logical data registers stay live throughout. Atomic Toffoli counts
    must be reported separately from any chosen T-equivalent conversion.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Controlled XOR model requires positive concrete width")
    root = ArithmeticCall("controlled_xor", (width,))
    gates = query_qec_gate_cost(
        root, declared_calls=arithmetic_call_graph(root), legacy_shims=legacy_shims
    )
    positive = ComponentCost(
        "ControlledXor",
        2 * width + 1,
        GateCost(**{f.name: getattr(gates, f.name) for f in fields(gates)}),
    )
    return controlled_component_cost(
        positive, registers, target_width=2 * width, legacy_shims=legacy_shims
    )
