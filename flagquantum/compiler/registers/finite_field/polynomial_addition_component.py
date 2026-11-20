# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2PolyAdd and GF2PolyAddK at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf_poly_arithmetic.

"""Compose descending typed coefficient additions through existing packed views."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .addition_component import field_addition_program
from .field_polynomial_addition import _constant_coefficients
from .field_polynomial_component import field_polynomial_component_program
from .field_polynomial_layout import FieldPolynomialLayout


def field_polynomial_addition_program(
    layout: FieldPolynomialLayout,
    coefficient: RegisterType,
    *,
    constants: tuple[int, ...] | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Add a polynomial word or encoded constant into descending coefficients.

    Packed f_x/g_x use the existing opaque view. Each coefficient retains the
    supplied binary-field descriptor. The packed polynomial dtype remains a
    separate admission boundary. This graph needs no workspace or measurements.
    Source quantum adjoints reverse coefficient calls while retaining each
    self-adjoint field XOR. Constant adjoints also reverse each child's X calls.
    """
    if type(adjoint) is not bool:
        raise ValueError("Polynomial addition direction requires a boolean")
    split = field_polynomial_component_program(layout, coefficient)
    join = field_polynomial_component_program(layout, coefficient, reverse=True)
    packed = opaque_register_type(layout.logical_width)
    ports = ("f_x", "g_x") if constants is None else ("f_x",)
    padded = None if constants is None else _constant_coefficients(constants, layout)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(**dict.fromkeys(ports, packed))
    )
    ordered_ports = tuple(reversed(ports)) if adjoint else ports
    words = {
        port: np.asarray(
            builder.call_program(split, {"reg": builder.inputs[port]})["reg"],
            dtype=object,
        )
        for port in ordered_ports
    }
    indices = tuple(range(layout.degree + 1))
    for i in reversed(indices) if adjoint else indices:
        constant = None if padded is None else padded[i]
        child = field_addition_program(coefficient, constant=constant, adjoint=adjoint)
        inputs = {"x": words["f_x"][i]}
        if padded is None:
            inputs["y"] = words["g_x"][i]
        result = builder.call_program(child, inputs)
        words["f_x"][i] = result["x"]
        if padded is None:
            words["g_x"][i] = result["y"]
    outputs = {
        port: builder.call_program(join, {"reg": words[port]})["reg"]
        for port in ordered_ports
    }
    return builder.finalize(outputs)
