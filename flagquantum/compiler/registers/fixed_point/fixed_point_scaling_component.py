# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 ScaleIntByReal supplies the signature and model.
# The root has no decomposition. Executable terms follow the existing D4 owner.

"""Typed real-times-integer scaling with an owned mixed fixed-point result."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import Register, RegisterType
from .truncated_product_component import truncated_product_program


def scale_integer_by_real_program(
    real_width: int, integer_width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve real_in/int_in and create or consume their truncated result.

    real_in is an unsigned pure fraction, int_in is unsigned, and result has
    real_width-integer_width fractional bits. All words are MSB first.
    The exact inverse consumes the computed image and returns its storage zero.
    """
    if any(
        type(width) is not int or width < 1 for width in (real_width, integer_width)
    ):
        raise ValueError("Fixed-point scaling requires positive concrete input widths")
    first_width = real_width - integer_width
    omitted_low_bits = 0
    if first_width < 0:
        # Every lower multiplier term shifts all real bits past the cutoff.
        omitted_low_bits = 1 - first_width
        first_width = 1
    return truncated_product_program(
        (
            Register("real_in", fixed_point_register_type(real_width, real_width)),
            Register("int_in", RegisterType("uint", integer_width)),
        ),
        fixed_point_register_type(real_width, real_width - integer_width),
        first_width,
        omitted_low_bits,
        adjoint=adjoint,
    )
