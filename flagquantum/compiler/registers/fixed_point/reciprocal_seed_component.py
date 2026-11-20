# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Qualtran v0.7.0 InvertRealNumber supplies the seed model without decomposition.
# Typed composition retains reciprocal_seed.py's Figure 4 construction.

"""Owned pure-fraction reciprocal seeds with stored multi-control cleanup."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.multi_control_component import multi_control_x_program
from .reciprocal import reciprocal_requirements


def reciprocal_seed_program(
    input_width: int, fraction: int, precision: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Create a zero seed's power-of-two estimate or consume its LEFT image.

    Source is unsigned mixed fixed point with a nonnegative fractional width.
    Seed is a pure fraction whose width covers the source integer bits. A zero
    integer part produces zero. The flag and multi-control work are private and
    restored before release. The inverse reverses selection before decomposition.
    """
    reciprocal_requirements(input_width, precision, source_fraction_bits=fraction)
    if type(adjoint) is not bool:
        raise ValueError("Reciprocal seed direction requires a boolean")
    source_dtype = fixed_point_register_type(input_width, fraction)
    seed_dtype = fixed_point_register_type(precision, precision)
    bit = quantum_bit_type()
    select = [
        Instruction("cx", (0, input_width + precision)),
        Instruction("cx", (0, input_width + input_width - fraction - 1)),
    ]
    integer_width = input_width - fraction
    flag_address = input_width + precision
    for index in range(1, integer_width):
        target = input_width + integer_width - 1 - index
        select.extend(
            (
                Instruction("x", (flag_address,)),
                Instruction("ccx", (index, flag_address, target)),
                Instruction("x", (flag_address,)),
                Instruction("cx", (target, flag_address)),
            )
        )
    selection = bit_gate_function(
        "seed_select",
        (Register("bits", bit, (input_width + precision + 1,)),),
        CircuitIR(
            input_width + precision + 1,
            tuple(reversed(select)) if adjoint else tuple(select),
        ),
    )
    flip = bit_gate_function(
        "seed_flip", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("a", source_dtype),
                Register(
                    "result",
                    seed_dtype,
                    side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
                ),
            )
        ),
        (selection, flip),
    )
    a = builder.inputs["a"]
    result = builder.inputs["result"] if adjoint else builder.allocate(dtype=seed_dtype)
    assert isinstance(a, RegisterBinding) and isinstance(result, RegisterBinding)
    source = list(builder.split(a))
    seed = list(builder.split(result))
    flag = builder.split(builder.allocate())[0]
    for action in (
        ("flip", "clear", "select") if adjoint else ("select", "clear", "flip")
    ):
        if action == "select":
            selected = builder.call_dict(
                "seed_select", {"bits": np.array([*source, *seed, flag], dtype=object)}
            )
            returned = list(np.asarray(selected["bits"]))
            source, seed, flag = (
                returned[:input_width],
                returned[input_width:-1],
                returned[-1],
            )
        elif action == "clear":
            cleared = builder.call_program(
                multi_control_x_program((0,) * integer_width),
                {
                    "controls": np.array(seed[:integer_width], dtype=object),
                    "target": flag,
                },
            )
            seed[:integer_width] = np.asarray(cleared["controls"])
            target_flag = cleared["target"]
            assert isinstance(target_flag, RegisterBinding)
            flag = target_flag
        else:
            flipped = builder.call_dict("seed_flip", {"q": flag})["q"]
            assert isinstance(flipped, RegisterBinding)
            flag = flipped
    builder.free(builder.join([flag]))
    seed_word = builder.join(seed, seed_dtype)
    returned_ports = {"a": builder.join(source, source_dtype)}
    if adjoint:
        builder.free(seed_word)
    else:
        returned_ports["result"] = seed_word
    return builder.finalize(returned_ports)
