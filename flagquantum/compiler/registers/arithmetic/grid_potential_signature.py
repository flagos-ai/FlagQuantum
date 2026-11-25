# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/potential.py.

"""Own the source pair's opaque three-axis ports without arithmetic claims."""

from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt


def grid_pair_potential_signature(width: SymbolicInt) -> Signature:
    """Retain the original ordered pair of three-axis THRU arrays."""
    word = opaque_register_type(width)
    return Signature(
        (Register("system_i", word, (3,)), Register("system_j", word, (3,)))
    )
