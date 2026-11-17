# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GFPolySplit and GFPolyJoin.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_poly_arithmetic/gf_poly_split_and_join.py

"""Share typed, gate-free rebinding for descending polynomial coefficients."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_graph import _namespace_helpers
from ..layout.component_partition import rebind_partition
from .field_polynomial_layout import FieldPolynomialLayout


def field_polynomial_component_program(
    layout: FieldPolynomialLayout,
    coefficient: RegisterType,
    name: str = "main",
    *,
    reverse: bool = False,
) -> ProgramRecord:
    """Bind a packed view and its typed coefficient array through the same wires."""
    if (
        coefficient.encoding != "finite_field"
        or executable_register_width(coefficient) != layout.coefficient_width
    ):
        raise ValueError(
            "polynomial coefficients require the matching executable field"
        )
    packed = Register(
        "reg", opaque_register_type(layout.logical_width), side=RegisterSide.LEFT
    )
    coefficients = Register(
        "reg", coefficient, (layout.degree + 1,), side=RegisterSide.RIGHT
    )
    if reverse:
        signature = Signature((coefficients.adjoint(), packed.adjoint()))
    else:
        signature = Signature((packed, coefficients))
    builder = ComponentProgramBuilder(signature, name=name)
    outputs = rebind_partition(builder, packed, (coefficients,), reverse=reverse)
    return _namespace_helpers(builder.finalize(outputs), name)
