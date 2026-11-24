# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiTargetCNOT.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_target_cnot.py

"""Retain the shared logarithmic-depth fanout sequence in typed calls."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .bitwise import fanout_ir


def fanout_component_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Control an opaque target word, MSB first, including arbitrary dirty targets.

    The control and target word remain caller-owned. No workspace, measurement
    or reset is introduced. Invalid or nonpositive concrete widths raise ValueError.
    """
    if type(adjoint) is not bool:
        raise ValueError("Fanout direction must be boolean")
    if type(width) is not int or width < 1:
        raise ValueError("Fanout requires a positive concrete integer width")
    bit = quantum_bit_type()
    word = bit if width == 1 else RegisterType("opaque", width)
    cx = bit_gate_function(
        "fanout_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(control=bit, targets=word), (cx,)
    )
    control, target = builder.inputs["control"], builder.inputs["targets"]
    assert isinstance(control, RegisterBinding) and isinstance(target, RegisterBinding)
    if width == 1:
        result = builder.call_dict(cx.name, {"ctrl": control, "target": target})
        return builder.finalize(
            {"control": result["ctrl"], "targets": result["target"]}
        )
    wires = [control, *builder.split(target)]
    for gate in fanout_ir(0, tuple(range(1, width + 1))):
        left, right = gate.wires
        result = builder.call_dict(
            cx.name, {"ctrl": wires[left], "target": wires[right]}
        )
        a, b = result["ctrl"], result["target"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        wires[left], wires[right] = a, b
    record = builder.finalize(
        {"control": wires[0], "targets": builder.join(wires[1:], word)}
    )
    leaf = ProgramRecord((cx,))
    return adjoint_component_calls(record, ((leaf, leaf),)) if adjoint else record
