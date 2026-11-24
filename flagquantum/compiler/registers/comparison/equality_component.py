# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 EqualsAConstant.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Retain integer views and literal multi-control equality calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program


def equality_constant_program(
    width: int, value: int, *, adjoint: bool = False
) -> ProgramRecord:
    """XOR equality into a dirty QBit, preserving the MSB-first unsigned input.

    The control component owns its clean predicate work and measurement erasure.
    Invalid widths or constants outside the represented domain raise ValueError.
    """
    if type(adjoint) is not bool:
        raise ValueError("Equality direction must be boolean")
    if (
        type(width) is not int
        or width < 1
        or type(value) is not int
        or not 0 <= value < 1 << width
    ):
        raise ValueError("Equality constant must fit a positive unsigned width")
    dtype = unsigned_register_type(width)
    child = multi_control_x_program(
        tuple((value >> i) & 1 for i in reversed(range(width)))
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype, target=quantum_bit_type())
    )
    word = builder.inputs["x"]
    assert isinstance(word, RegisterBinding)
    result = builder.call_program(
        child, {"controls": builder.split(word), "target": builder.inputs["target"]}
    )
    record = builder.finalize(
        {"x": builder.join(result["controls"], dtype), "target": result["target"]}
    )
    return adjoint_component_calls(record, ((child, child),)) if adjoint else record


def equality_program(dtype: RegisterType) -> ProgramRecord:
    """XOR equality of two encoded words, retaining Xor/MCX/Xor source calls."""
    width = executable_register_width(dtype)
    if width < 1:
        raise ValueError("Equality requires a positive concrete register width")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype, y=dtype, target=quantum_bit_type())
    )
    xor = bitwise_word_program(dtype, "xor")
    result = builder.call_program(
        xor, {"x": builder.inputs["x"], "y": builder.inputs["y"]}
    )
    word = result["y"]
    assert isinstance(word, RegisterBinding)
    compared = builder.call_program(
        multi_control_x_program((0,) * width),
        {"controls": builder.split(word), "target": builder.inputs["target"]},
    )
    restored = builder.call_program(
        xor, {"x": result["x"], "y": builder.join(compared["controls"], dtype)}
    )
    return builder.finalize({**restored, "target": compared["target"]})
