# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2PolyAdd and GF2PolyAddK at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf_poly_arithmetic.

"""Compose descending typed coefficient additions through existing packed views."""

from collections.abc import Callable

import numpy as np

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .addition_component import field_addition_program
from .field_polynomial_addition import _constant_coefficients
from .field_polynomial_component import (
    field_polynomial_component_declaration,
    field_polynomial_component_program,
)
from .field_polynomial_layout import FieldPolynomialLayout


def field_polynomial_addition_program(
    layout: FieldPolynomialLayout,
    coefficient: RegisterType,
    *,
    constants: tuple[int, ...] | None = None,
    adjoint: bool = False,
    packed_dtype: RegisterType | None = None,
) -> ProgramRecord:
    """Add a polynomial word or encoded constant into descending coefficients.

    Packed f_x/g_x accept the explicit canonical polynomial descriptor approved
    by Proposal 109. The default retains the existing opaque view. Coefficients
    retain their binary-field identity. Workspace and measurements are zero.
    Source quantum adjoints reverse coefficient calls while retaining each
    self-adjoint field XOR. Constant adjoints also reverse each child's X calls.
    """
    if type(adjoint) is not bool:
        raise ValueError("Polynomial addition direction requires a boolean")
    split = field_polynomial_component_program(
        layout, coefficient, packed_dtype=packed_dtype
    )
    join = field_polynomial_component_program(
        layout, coefficient, reverse=True, packed_dtype=packed_dtype
    )
    packed = (
        opaque_register_type(layout.logical_width)
        if packed_dtype is None
        else packed_dtype
    )
    padded = None if constants is None else _constant_coefficients(constants, layout)
    return _polynomial_addition_root(
        layout.degree,
        packed,
        split,
        join,
        lambda constant: field_addition_program(
            coefficient, constant=constant, adjoint=adjoint
        ),
        padded,
        adjoint,
    )


def logical_field_polynomial_addition_program(
    degree: int,
    coefficient: RegisterType,
    packed: RegisterType,
    *,
    constants: tuple[int, ...] | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain finite source calls over a symbolic binary coefficient field.

    Atomic views and unresolved field additions are declarations without bodies.
    This supplies neither physical gates nor executable zero-cost identities.
    The caller owns constant coefficient evaluation; no field values enter here.
    """
    if type(degree) is not int or degree < 0 or type(adjoint) is not bool:
        raise ValueError(
            "Logical polynomial calls require a concrete degree and direction"
        )
    parameters = dict(coefficient.parameters)
    if (
        coefficient.encoding != "finite_field"
        or tuple(parameters) != ("characteristic", "degree", "order")
        or parameters["characteristic"] != 2
        or coefficient.num_qubits != parameters["degree"]
    ):
        raise ValueError(
            "Source polynomial children require their default binary field"
        )
    padded = None
    if constants is not None:
        if (
            not constants
            or any(constants[: -(degree + 1)])
            or any(type(v) is not int or v < 0 for v in constants)
        ):
            raise ValueError(
                "Logical polynomial constants require nonnegative coefficient words"
            )
        trimmed = constants[-(degree + 1) :]
        padded = (0,) * (degree + 1 - len(trimmed)) + trimmed
    split = field_polynomial_component_declaration(
        degree, coefficient, "polynomial_split", packed_dtype=packed
    )
    join = field_polynomial_component_declaration(
        degree, coefficient, "polynomial_join", reverse=True, packed_dtype=packed
    )

    def addition(constant: int | None) -> ProgramRecord:
        name = "field_add" if constant is None else f"field_add_constant_{constant}"
        if constant is not None and adjoint:
            name += "_adjoint"
        ports = Signature.build_from_dtypes(
            **dict.fromkeys(("x", "y") if constant is None else ("x",), coefficient)
        )
        return ProgramRecord((FunctionRecord(name, (), registers=tuple(ports)),))

    return _polynomial_addition_root(
        degree, packed, split, join, addition, padded, adjoint
    )


def _polynomial_addition_root(
    degree: int,
    packed: RegisterType,
    split: ProgramRecord,
    join: ProgramRecord,
    addition: Callable[[int | None], ProgramRecord],
    padded: tuple[int, ...] | None,
    adjoint: bool,
) -> ProgramRecord:
    ports = ("f_x", "g_x") if padded is None else ("f_x",)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(**dict.fromkeys(ports, packed))
    )
    ordered_ports = tuple(reversed(ports)) if adjoint else ports
    words = {
        port: np.asarray(
            builder.call_program(
                split, {"reg": builder.inputs[port]}, entry=split.functions[0].name
            )["reg"],
            dtype=object,
        )
        for port in ordered_ports
    }
    indices = tuple(range(degree + 1))
    for i in reversed(indices) if adjoint else indices:
        constant = None if padded is None else padded[i]
        child = addition(constant)
        inputs = {"x": words["f_x"][i]}
        if padded is None:
            inputs["y"] = words["g_x"][i]
        result = builder.call_program(child, inputs, entry=child.functions[0].name)
        words["f_x"][i] = result["x"]
        if padded is None:
            words["g_x"][i] = result["y"]
    outputs = {
        port: builder.call_program(
            join, {"reg": words[port]}, entry=join.functions[0].name
        )["reg"]
        for port in ordered_ports
    }
    return builder.finalize(outputs)
