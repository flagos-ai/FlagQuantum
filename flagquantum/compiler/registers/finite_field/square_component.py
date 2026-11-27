# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Decomposition schedule ported from locked Qualtran v0.7.0 GF2Square.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_square.py

"""Preserve the frozen square decomposition separately from exact Frobenius."""

from operator import index
from typing import SupportsIndex

from ....core._binary_matrix import BinaryMatrix
from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import RegisterType
from .field_linear import _frobenius_matrix
from .field_multiplication import _validate_field
from .field_word import binary_field_word_width
from .linear_component import field_linear_program


def field_square_decomposition_matrix(
    width: int, count: SupportsIndex, *, modulus: int
) -> tuple[tuple[int, ...], ...]:
    """Compile the original integer matrix-power schedule without correcting it.

    Rows and columns are least-significant coefficient first. The initial
    polynomial square is shared with the exact field arithmetic owner. The
    remaining loop retains the locked M/R updates, including its zero/negative
    count and non-power-of-two behavior; it is not a mathematical power API.
    """
    _validate_field(width, modulus)
    try:
        exponent = index(count)
    except TypeError as error:
        raise ValueError(
            "Square decomposition count requires a concrete integer"
        ) from error
    matrix = BinaryMatrix.from_dense(_frobenius_matrix(width, 1, modulus))
    result = BinaryMatrix.identity(width)
    while exponent > 1:
        if exponent & 1:
            result = result.multiply(result)
        matrix = matrix.multiply(matrix)
        exponent >>= 1
    return tuple(tuple(row) for row in matrix.multiply(result).dense())


def field_square_decomposition_program(
    dtype: RegisterType,
    count: SupportsIndex = 1,
    *,
    modulus: int,
    uncompute: bool = False,
) -> ProgramRecord:
    """Store the actual original Split/linear/Join execution, including defects.

    The source toggles the linear child's declared inverse flag, but its
    physical builder ignores it. Both directions here therefore use the same
    primitive recipe and aliases. Exact mathematical powers and inverses retain
    their separate field_frobenius_program owner. No work or measurements arise.
    """
    if type(uncompute) is not bool:
        raise ValueError("Square decomposition direction requires a boolean")
    width = binary_field_word_width(dtype)
    if dict(dtype.parameters).get("polynomial", modulus) != modulus:
        raise ValueError("Square decomposition modulus must match the descriptor")
    matrix = field_square_decomposition_matrix(width, count, modulus=modulus)
    return field_linear_program(dtype, matrix)
