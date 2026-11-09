"""Shape-level costs of the two Hubbard SELECT consumers."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's HubbardMajorannaOperator and HubbardSpinUpZ.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py

from __future__ import annotations

from typing import cast

from ...compiler._component_trace.symbolic_math import SymbolicInt, ceil, log2, smax
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)


def _leaf_width(side: SymbolicInt, controlled: bool, accumulator: int) -> SymbolicInt:
    controls = int(controlled)
    if isinstance(side, int):
        if type(side) is not int or side < 2:
            raise ValueError(
                "Hubbard resource formulas require side length at least two"
            )
        address = 2 * (side - 1).bit_length()
        return 2 * side**2 + 2 * address + 1 + 2 * controls + accumulator

    # The source cannot decompose symbolic lattice widths. QubitCount takes
    # the maximum of the signature width and the largest direct callee (And).
    ports = 2 * side**2 + 2 * ceil(log2(side)) + 1 + controls
    return cast(SymbolicInt, smax(3, ports))


def source_hubbard_majorana_formula(
    side: SymbolicInt, *, controlled: bool = False
) -> SymbolicResourceFormula:
    """Count one source Majorana SELECT in either the X or Y basis."""
    width = _leaf_width(side, controlled, accumulator=1)
    ands = 2 * side**2 - 2 + int(controlled)
    return SymbolicResourceFormula(ands, ands, ands, width)


def source_hubbard_spin_up_formula(
    side: SymbolicInt, *, controlled: bool = False
) -> SymbolicResourceFormula:
    """Count the source spin-up Z and its address predicate traversal."""
    width = _leaf_width(side, controlled, accumulator=0)
    sites = side**2
    ands = sites - 1 + int(controlled)
    return SymbolicResourceFormula(ands, ands + 2 * sites - 1, ands, width)
