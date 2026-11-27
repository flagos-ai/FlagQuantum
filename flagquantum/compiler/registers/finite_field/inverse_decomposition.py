# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ported from Qualtran v0.7.0 GF2Inverse's degree-one branch.
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
    """Build validated literal source profiles, not the mathematical inverse.

    MSB-first x is THRU; result and retained logical junk are RIGHT. Recursive
    padding is private clean work. The square and product dependencies retain
    every original declared direction, including the ignored linear adjoint.

    Profiles with a root temporary release require a separate clean-release
    proof: the original four/eight-bit paths release nonzero work. They are
    refused here rather than weakening clean-free semantics. This restriction
    is not a claim that every other unvalidated degree is a source defect.
    """
    if dict(dtype.parameters).get("degree") == 1:
        return field_inverse_degree_one_program(dtype)
    width = field_word_width(dtype, modulus)
    if (width - 1).bit_count() > 1:
        raise ValueError(
            "Literal inverse root temporary release has no clean-work proof; "
            "the original degree-four/eight paths release nonzero work"
        )
    return field_inverse_decomposition_graph(dtype, modulus=modulus)


def field_inverse_decomposition_graph(
    dtype: RegisterType, *, modulus: int
) -> ProgramRecord:
    """Retain the entire static original graph, including unproved clean frees.

    This is a presentation/source-audit artifact, not execution admission.
    Use field_inverse_decomposition_program for the guarded literal execution
    profile or field_inverse_program for the correct mathematical operation.
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
