# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwap.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Retain register exchange calls and the existing exact seven-T bit recipe."""

from dataclasses import replace

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_control import control_component_program
from ..component_gate_function import bit_gate_function
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from ..control.specialized_component_control import specialized_component_control_system
from .controlled_swap import controlled_swap_bit_ir


def positive_controlled_swap_program(width: int) -> ProgramRecord:
    """Control the source word CSwap through its packed bit specializations.

    The outer ctrl2 and original ctrl precede MSB-first x/y words. Split/Join
    pass the outer control unchanged; each bit exchange owns one restored
    conjunction and one measurement through the shared specialization owner.
    """
    record = controlled_swap_program(width)
    leaf = next(
        function for function in record.functions if function.name == "swap_bit"
    )
    positive = replace(leaf, name="positive_swap_bit")
    module, adder = specialized_component_control_system(
        ProgramRecord((positive,)), "ctrl", 1, 1, positive.name
    )
    used = {operation.callee for operation in record.functions[0].blocks[0].operations}
    controls = {
        function.name: (
            adder
            if function.name == leaf.name
            else bookkeeping_scalar_control_adder(function)
        )
        for function in record.functions[1:]
        if function.name in used
    }
    return control_component_program(
        replace(record, functions=(*record.functions, *module.functions)), controls
    )


def controlled_swap_program(width: int) -> ProgramRecord:
    """Exchange two opaque MSB-first words under a coherent through control.

    The self-adjoint source bit calls use the exact seven-T recipe in both
    directions. No work, reset or measurement is introduced.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Controlled swap requires a positive concrete width")
    bit = quantum_bit_type()
    word = bit if width == 1 else RegisterType("opaque", width)
    leaf = bit_gate_function(
        "swap_bit",
        (Register("ctrl", bit), Register("x", bit), Register("y", bit)),
        CircuitIR(3, controlled_swap_bit_ir(0, 1, 2)),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=bit, x=word, y=word), (leaf,)
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xs, ys, ctrl = builder.split(x), builder.split(y), builder.inputs["ctrl"]
    for index in range(width):
        result = builder.call_dict(
            "swap_bit", {"ctrl": ctrl, "x": xs[index], "y": ys[index]}
        )
        ctrl, xs[index], ys[index] = result["ctrl"], result["x"], result["y"]
    return builder.finalize(
        {"ctrl": ctrl, "x": builder.join(xs, word), "y": builder.join(ys, word)}
    )
