# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Square at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_square.py.

"""Typed Frobenius graphs sharing the existing mathematical matrix authority."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import RegisterType
from .field_linear import _frobenius_matrix
from .field_multiplication import _validate_field
from .field_word import binary_field_word_width
from .linear_component import field_linear_program


def field_frobenius_program(
    dtype: RegisterType, count: int = 1, *, modulus: int, adjoint: bool = False
) -> ProgramRecord:
    """Raise an MSB-first field word x to 2**count, or apply its exact inverse.

    One stored linear-map call operates on reversed coefficient bits. The dtype
    and polynomial must describe the same binary field. Logical output aliases
    carry the free permutation. No additional quantum or classical work is used.
    Source matrix-power and ignored-adjoint defects remain separate witnesses.
    """
    if dtype.encoding != "finite_field":
        raise ValueError("Frobenius requires a binary field descriptor")
    width = binary_field_word_width(dtype)
    _validate_field(width, modulus)
    parameters = dict(dtype.parameters)
    if parameters.get("polynomial", modulus) != modulus:
        raise ValueError("Frobenius modulus must match the field descriptor")
    if type(count) is not int or count < 0:
        raise ValueError("Frobenius count must be a nonnegative concrete integer")
    matrix = tuple(tuple(row) for row in _frobenius_matrix(width, count, modulus))
    return field_linear_program(dtype, matrix, adjoint=adjoint)
