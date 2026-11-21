# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OutOfPlaceAdder.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Own the sum output and consume its exact measured adjoint in typed calls."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_gate_function import bit_gate_function
from ..component_wire_graph import WireCall, wire_component_program
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
    word, bit = unsigned_register_type(width), quantum_bit_type()
    output = unsigned_register_type(count)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    conjunction = and_component_program((1, 1), name="sum_and", uncompute=adjoint)
    cnot = ProgramRecord(
        (
            bit_gate_function(
                "sum_cx",
                (Register("ctrl", bit), Register("target", bit)),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            ),
        )
    )
    signature = Signature(
        (Register("a", word), Register("b", word), Register("c", output, side=side))
    )
    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    targets = tuple(range(2 * width, 2 * width + count))
    calls: list[WireCall] = []
    for kind, operands in out_of_place_addition_components(
        left, right, targets, include_msb=include_msb, adjoint=adjoint
    ):
        x, y = operands[:2]
        calls.append(
            (cnot, {"ctrl": (x,), "target": (y,)})
            if kind == "cx"
            else (conjunction, {"ctrl": (x, y), "target": (operands[2],)})
        )
    ports, sum_port = {"a": left, "b": right}, {"c": targets}
    return wire_component_program(
        signature,
        {**ports, **(sum_port if adjoint else {})},
        {**ports, **({} if adjoint else sum_port)},
        calls,
    )
