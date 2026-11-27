# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GFPolySplit and GFPolyJoin.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_poly_arithmetic/gf_poly_split_and_join.py

"""Share typed, gate-free rebinding for descending polynomial coefficients."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.register_verifier import (
    executable_register_width,
    logical_register_width,
)
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
    packed_dtype: RegisterType | None = None,
) -> ProgramRecord:
    """Bind a packed view and its typed coefficient array through the same wires."""
    if (
        coefficient.encoding != "finite_field"
        or executable_register_width(coefficient) != layout.coefficient_width
    ):
        raise ValueError(
            "polynomial coefficients require the matching executable field"
        )
    if packed_dtype is not None:
        expected = RegisterType(
            "finite_field_polynomial",
            layout.logical_width,
            parameters=(("polynomial_degree", layout.degree),) + coefficient.parameters,
        )
        if packed_dtype != expected:
            raise ValueError(
                "Packed polynomial and coefficient field metadata must match"
            )
        executable_register_width(packed_dtype)
    packed = Register(
        "reg",
        (
            opaque_register_type(layout.logical_width)
            if packed_dtype is None
            else packed_dtype
        ),
        side=RegisterSide.LEFT,
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


def field_polynomial_component_declaration(
    degree: int,
    coefficient: RegisterType,
    name: str = "main",
    *,
    reverse: bool = False,
    packed_dtype: RegisterType | None = None,
) -> ProgramRecord:
    """Declare the original atomic view without assuming an executable width.

    Concrete extent retains source constructor admission. The coefficient field
    and packed polynomial metadata remain intact. Empty blocks are declarations,
    not the executable identities produced by the physical view builder above.
    """
    if type(degree) is not int or degree < 0 or type(reverse) is not bool:
        raise ValueError(
            "Polynomial view declarations require a concrete degree and direction"
        )
    if coefficient.encoding != "finite_field":
        raise ValueError("Polynomial view coefficients require a field descriptor")
    width = logical_register_width(coefficient)
    expected = RegisterType(
        "finite_field_polynomial",
        width * (degree + 1),
        parameters=(("polynomial_degree", degree),) + coefficient.parameters,
    )
    packed = expected if packed_dtype is None else packed_dtype
    if packed != expected:
        raise ValueError("Packed polynomial and coefficient declarations must agree")
    logical_register_width(packed)
    word = Register("reg", packed, side=RegisterSide.LEFT)
    array = Register("reg", coefficient, (degree + 1,), RegisterSide.RIGHT)
    ports = (array.adjoint(), word.adjoint()) if reverse else (word, array)
    return ProgramRecord((FunctionRecord(name, (), registers=ports),))
