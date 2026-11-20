# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 multiplication.Product supplies the sided register/model contract.
# The frozen root supplies no executable decomposition.

"""Owned integer-product results over the shared column-compression graph."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterType
from .column_result_component import column_result_program
from .integer_product import _product_terms, integer_product_requirements


def integer_product_program(
    left_width: int, right_width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve THRU a/b and create RIGHT product, or consume its LEFT image.

    Inputs are unsigned MSB-first words. Result width is twice the larger input
    width. The inverse requires the coherent matching image and releases its
    zeroed storage. Shared columns own and restore product/carry scratch.
    """
    result_width, workspace_width = integer_product_requirements(
        left_width, right_width
    )
    if type(adjoint) is not bool:
        raise ValueError("Integer product direction requires a boolean")
    return column_result_program(
        (
            Register("a", RegisterType("uint", left_width)),
            Register("b", RegisterType("uint", right_width)),
        ),
        result_width,
        workspace_width,
        tuple(
            _product_terms(
                tuple(range(left_width)),
                tuple(range(left_width, left_width + right_width)),
            )
        ),
        adjoint=adjoint,
    )
