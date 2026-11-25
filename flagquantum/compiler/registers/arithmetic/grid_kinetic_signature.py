# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/kinetic.py.

"""Retain original grid kinetic storage independently of numerical execution."""

from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt


def grid_coordinate_width(grid_points: int) -> int:
    """Keep the source integer bit-length rule in one storage owner."""
    return (grid_points - 1).bit_length() + 1


def grid_kinetic_signature(electrons: SymbolicInt, grid_points: int) -> Signature:
    """Declare the original electron/axis array and integer bit-length sizing."""
    width = grid_coordinate_width(grid_points)
    return Signature(
        (Register("system", opaque_register_type(width), shape=(electrons, 3)),)
    )
