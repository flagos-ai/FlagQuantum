# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 chemistry/trotter/grid_ham/inverse_sqrt.py.

"""Own opaque source ports without assigning an arithmetic representation."""

from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt


def grid_inverse_sqrt_signature(
    distance_width: SymbolicInt,
    polynomial_width: SymbolicInt,
    target_width: SymbolicInt,
    *,
    newton: bool,
) -> Signature:
    """Retain the original three THRU ports and polynomial coefficient array."""
    return Signature(
        (
            Register("x_sq", opaque_register_type(distance_width)),
            Register(
                "poly" if newton else "in_coeff",
                opaque_register_type(polynomial_width),
                shape=() if newton else (4,),
            ),
            Register("target" if newton else "out", opaque_register_type(target_width)),
        )
    )
