# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 multiplication.Square supplies the sided numeric contract.
# The frozen root supplies no executable decomposition.

"""Typed integer square with shared measured column-compression calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterType
from .column_result_component import column_result_program
from .integer_square import _square_terms, integer_square_workspace_size


def integer_square_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Create a RIGHT square result or erase its matching LEFT image.

    Unsigned words are MSB first. Forward result storage starts clean. The
    inverse requires the coherent square image and releases its zeroed result.
    The column owner allocates width*(width-1) product/carry bits and restores
    them through shared And compute and measured erasure calls.
    """
    count = integer_square_workspace_size(width)
    if type(adjoint) is not bool:
        raise ValueError("Integer square direction requires a boolean")
    return column_result_program(
        (Register("a", RegisterType("uint", width)),),
        2 * width,
        count,
        tuple(
            (column, width - 1 - i, width - 1 - j if i != j else None)
            for i, j, column in _square_terms(width)
        ),
        adjoint=adjoint,
    )
