# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 multiplication.SumOfSquares supplies the shaped signature/model.
# The frozen root supplies no executable decomposition.

"""Shaped sum-of-squares results through shared measured column calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterType
from .column_result_component import column_result_program
from .integer_square import _square_terms, sum_of_squares_requirements


def sum_of_squares_program(
    width: int, count: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve a row-major input array and create or erase its unsigned sum.

    Every MSB-first input word has width bits. Result storage has
    2*width + (count-1).bit_length() bits, and is RIGHT for compute or LEFT for
    erasure. The inverse requires the coherent image. All scratch returns clean.
    """
    result_width, workspace_width = sum_of_squares_requirements(width, count)
    if type(adjoint) is not bool:
        raise ValueError("Sum of squares direction requires a boolean")
    terms = tuple(
        (
            column,
            row * width + width - 1 - i,
            row * width + width - 1 - j if i != j else None,
        )
        for row in range(count)
        for i, j, column in _square_terms(width)
    )
    return column_result_program(
        (Register("input", RegisterType("uint", width), (count,)),),
        result_width,
        workspace_width,
        terms,
        adjoint=adjoint,
    )
