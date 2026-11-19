# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OutOfPlaceAdder.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Own the sum output and consume its exact measured adjoint in typed calls."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.conjunction_program import and_component_program
from .out_of_place_addition_structure import out_of_place_addition_components


def out_of_place_addition_program(
    width: int,
    *,
    include_msb: bool = True,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain unsigned a/b and create their sum c, or consume a correct sum.

    Words are MSB first. c has width+1 bits with the carry, otherwise width.
    The forward graph owns a clean low output bit and each And output. The
    adjoint measures and clears computed And outputs and frees the low bit.
    Binding a newly created output explicitly transfers workspace ownership.
    Binding a separately supplied c for erasure requires input ownership.
    Invalid dimensions or flags raise ValueError.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Out-of-place addition width must be a positive integer")
    if type(include_msb) is not bool or type(adjoint) is not bool:
        raise ValueError("Out-of-place addition flags require booleans")
    count = width + int(include_msb)
    word, bit = RegisterType("uint", width), quantum_bit_type()
    output = RegisterType("uint", count)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    conjunction = and_component_program((1, 1), name="sum_and", uncompute=adjoint)
    cnot = bit_gate_function(
        "sum_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature(
            (Register("a", word), Register("b", word), Register("c", output, side=side))
        ),
        (cnot, *conjunction.functions),
    )
    a, b = builder.inputs["a"], builder.inputs["b"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    targets = tuple(range(2 * width, 2 * width + count))
    wires = dict(enumerate((*builder.split(a), *builder.split(b))))
    if adjoint:
        c = builder.inputs["c"]
        assert isinstance(c, RegisterBinding)
        wires.update(zip(targets, builder.split(c), strict=True))
    else:
        wires[targets[-1]] = builder.allocate(dtype=bit)
    for kind, operands in out_of_place_addition_components(
        left, right, targets, include_msb=include_msb, adjoint=adjoint
    ):
        x, y = operands[:2]
        if kind == "cx":
            result = builder.call_dict("sum_cx", {"ctrl": wires[x], "target": wires[y]})
            ctrl, target = result["ctrl"], result["target"]
            assert isinstance(ctrl, RegisterBinding) and isinstance(
                target, RegisterBinding
            )
            wires[x], wires[y] = ctrl, target
            continue
        values = {"ctrl": np.array([wires[x], wires[y]], dtype=object)}
        if adjoint:
            result = builder.call_dict(
                "sum_and", {**values, "target": wires.pop(operands[2])}
            )
        else:
            result = builder.call_dict("sum_and", values)
            target = result["target"]
            assert isinstance(target, RegisterBinding)
            wires[operands[2]] = target
        wires[x], wires[y] = np.asarray(result["ctrl"])
    if adjoint:
        builder.free(wires.pop(targets[-1]))
    outputs = {
        "a": builder.join([wires[w] for w in left], word),
        "b": builder.join([wires[w] for w in right], word),
    }
    if not adjoint:
        outputs["c"] = builder.join([wires[w] for w in targets], output)
    return builder.finalize(outputs)
