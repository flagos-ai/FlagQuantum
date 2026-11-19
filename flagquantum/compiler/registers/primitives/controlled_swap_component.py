# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwap.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Retain register exchange calls and the existing exact seven-T bit recipe."""

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .controlled_swap import controlled_swap_bit_ir


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
