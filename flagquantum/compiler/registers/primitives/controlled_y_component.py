# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Follows locked Qualtran v0.7.0 CirqGateAsBloq CY decomposition and controls.

"""Retain the three source calls of a generically controlled Cirq CY view."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.specialized_component_control import multi_controlled_component_program
from .pauli_rotation import pauli_power_ir


def controlled_scalar_y_program() -> ProgramRecord:
    """Retain scalar CYGate's two-control packed helper and measured cleanup.

    ctrl is the ordered two-bit control array and target is one THRU bit.
    Unlike the Cirq q-array view, CYGate specializes through its existing
    positive CY body. The shared helper owns one restored private predicate.
    """
    positive = ProgramRecord(
        (
            bit_gate_function(
                "main",
                tuple(Signature.build(ctrl=1, target=1)),
                CircuitIR(2, (Instruction("cy", (0, 1)),)),
            ),
        )
    )
    return multi_controlled_component_program(positive, "ctrl", (1, 1))


def controlled_cirq_y_program() -> ProgramRecord:
    """Control the source q-array CY or inverse-CY through its original calls.

    Ports are outer scalar ctrl followed by the two-bit q array (old control,
    target). Both source directions use positive half-X, packed-controlled CZ,
    negative half-X. Half powers are source atomic leaves; their exact native
    realization reuses Pauli-power phase conventions. The shared CZ helper
    owns one clean predicate and measured erasure. No SDK import is needed.
    This owner describes the Cirq q-array view, not the scalar CYGate class.
    It follows Cirq 1.7's CY view. Cirq 1.6's generic ControlledGate(Y) has a
    different six-call source recipe and must not be bound to this owner.
    """
    bit = quantum_bit_type()
    positive = bit_gate_function(
        "positive_half_x",
        tuple(Signature.build(ctrl=1, q=1)),
        CircuitIR(2, pauli_power_ir("x", 1, 0.5, control=0)),
    )
    negative = bit_gate_function(
        "negative_half_x",
        positive.registers or (),
        CircuitIR(2, pauli_power_ir("x", 1, -0.5, control=0)),
    )
    cz = ProgramRecord(
        (
            bit_gate_function(
                "main",
                tuple(Signature.build(q1=1, q2=1)),
                CircuitIR(2, (Instruction("cz", (0, 1)),)),
            ),
        )
    )
    packed = multi_controlled_component_program(cz, "q1", (1, 1))
    builder = ComponentProgramBuilder(
        Signature((Register("ctrl", bit), Register("q", bit, (2,)))),
        (positive, negative),
    )
    ctrl, q = builder.inputs["ctrl"], np.asarray(builder.inputs["q"])
    first = builder.call_dict(positive.name, {"ctrl": ctrl, "q": q[1]})
    middle = builder.call_program(
        packed,
        {"q1": np.array([first["ctrl"], q[0]], dtype=object), "q2": first["q"]},
    )
    controls = np.asarray(middle["q1"])
    last = builder.call_dict(negative.name, {"ctrl": controls[0], "q": middle["q2"]})
    return builder.finalize(
        {"ctrl": last["ctrl"], "q": np.array([controls[1], last["q"]], dtype=object)}
    )
