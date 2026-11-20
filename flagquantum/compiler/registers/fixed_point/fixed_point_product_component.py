# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 MultiplyTwoReals supplies the signature and model.
# The root has no decomposition. Executable terms follow the existing D5 owner.

"""Typed pure-fraction multiplication using shared truncated-product calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import Register
from .fixed_point_scaling import fixed_point_product_workspace_size
from .truncated_product_component import truncated_product_program


def fixed_point_product_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Create the pure-fraction product or consume its coherent LEFT image.

    a, b and result retain the original unsigned pure-fraction descriptors and
    MSB-first order. The lowest multiplier bit contributes no retained term.
    Shared CAdd restores owned work. Erase expects the computed result image.
    """
    fixed_point_product_workspace_size(width)
    dtype = fixed_point_register_type(width, width)
    return truncated_product_program(
        (Register("a", dtype), Register("b", dtype)),
        dtype,
        1,
        1,
        adjoint=adjoint,
    )
