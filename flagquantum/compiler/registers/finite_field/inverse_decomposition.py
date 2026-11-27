# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ported from Qualtran v0.7.0 GF2Inverse's degree-one branch.
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_inverse.py.

"""Literal degree-one inverse branch on existing raw field storage."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .addition_component import field_addition_decomposition_program
from .field_word import raw_field_word_width


def field_inverse_degree_one_program(dtype: RegisterType) -> ProgramRecord:
    """Allocate a RIGHT result and apply the source's degree-prefix addition.

    Raw width may exceed degree one. No numerical inverse claim is made outside
    binary fields. There is no junk port, private work or measured branch.
    """
    raw_field_word_width(dtype)
    if dict(dtype.parameters).get("degree") != 1:
        raise ValueError("This inverse source branch requires field degree one")
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", dtype),
                Register("result", dtype, side=RegisterSide.RIGHT),
            )
        )
    )
    outputs = builder.call_program(
        field_addition_decomposition_program(dtype),
        {"x": builder.inputs["x"], "y": builder.allocate(dtype=dtype)},
    )
    return builder.finalize({"x": outputs["x"], "result": outputs["y"]})
