# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 EqualsAConstant.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Retain integer views and literal multi-control equality calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program


def equality_constant_program(width: int, value: int) -> ProgramRecord:
    """XOR equality into a dirty QBit, preserving the MSB-first unsigned input.

    The control component owns its clean predicate work and measurement erasure.
    Invalid widths or constants outside the represented domain raise ValueError.
    """
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
    return builder.finalize(
        {"x": builder.join(result["controls"], dtype), "target": result["target"]}
    )
