# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 SquareRealNumber supplies the dual-input register signature.
# The frozen root has no decomposition. D24/D28 terms come from the existing
# paper construction in fixed_point_square.py.

"""Typed fractional square retains both source input ports and owned output."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import Register
from ..arithmetic.column_result_component import column_result_program
from .fixed_point_square import _retained_terms, fixed_point_square_workspace_size


def fixed_point_square_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Create the fractional square image or consume its matching LEFT image.

    Both THRU inputs retain the original unsigned pure-fraction dtype and
    MSB-first order. The paper construction reads a and preserves b. Result
    storage starts zero for compute. Erase consumes its coherent image and
    releases the restored output. Shared columns own measured scratch cleanup.
    """
    work = fixed_point_square_workspace_size(width)
    if type(adjoint) is not bool:
        raise ValueError("Fixed-point square direction requires a boolean")
    dtype = fixed_point_register_type(width, width)
    return column_result_program(
        (Register("a", dtype), Register("b", dtype)),
        width,
        work,
        tuple(
            (column, high - 1, low - 1 if low is not None else None)
            for column, high, low in _retained_terms(width)
        ),
        adjoint=adjoint,
        result_dtype=dtype,
    )
