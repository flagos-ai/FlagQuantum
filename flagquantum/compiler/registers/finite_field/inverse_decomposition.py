# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal dependencies adapted from Qualtran v0.7.0 GF2Inverse.
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_inverse.py.

"""Literal inverse dependencies with explicit source clean-release boundaries."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .addition_component import field_addition_decomposition_program
from .field_word import field_word_width, raw_field_word_width
from .inverse_component import _field_inverse_schedule
from .karatsuba_component import karatsuba_decomposition_program
from .square_component import field_square_decomposition_program


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


def field_inverse_decomposition_program(
    dtype: RegisterType, *, modulus: int
) -> ProgramRecord:
    """Retain the entire static original inverse graph and its preconditions.

    All words are MSB-first; x is THRU and result/junk are RIGHT. This includes
    original invalid clean releases, not permission to execute them. The append
    consumer verifies actual bound releases with Simulation's exact Boolean
    analysis; mathematical field_inverse_program remains independent.
    """
    if dict(dtype.parameters).get("degree") == 1:
        return field_inverse_degree_one_program(dtype)
    width = field_word_width(dtype, modulus)
    return _field_inverse_schedule(
        dtype,
        width,
        karatsuba_decomposition_program(dtype, modulus=modulus),
        lambda count, adjoint: field_square_decomposition_program(
            dtype, count, modulus=modulus, uncompute=adjoint
        ),
    )
