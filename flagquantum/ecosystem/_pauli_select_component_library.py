# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata follows locked Qualtran v0.7.0 SelectPauliLCU.nth_operation.

"""Capture signed source Pauli data without its object-array wire codec."""

from math import isfinite
from typing import Any, cast

from ..compiler.registers.lookup.pauli_select_component import (
    pauli_select_component_pair,
)
from ..core.ir.program import ProgramRecord
from ..errors import SerializationError


def capture_pauli_select_pair(source: object) -> tuple[ProgramRecord, ProgramRecord]:
    """Replace source payload immediately, retaining its sign-of-real convention."""
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU

    if not isinstance(source, SelectPauliLCU):
        raise SerializationError("Expected an explicit source Pauli SELECT")
    foreign = cast(Any, source)
    words: list[str] = []
    for pauli in foreign.select_unitaries:
        real = complex(pauli.coefficient).real
        if not isfinite(real) or real == 0:
            raise SerializationError("Source sign-of-real Pauli leaf is not unitary")
        letters = "".join("IXYZ"[int(letter)] for letter in pauli.pauli_mask)
        words.append(("-" if real < 0 else "") + letters)
    return pauli_select_component_pair(
        foreign.selection_bitsize,
        foreign.target_bitsize,
        tuple(words),
        control_value=foreign.control_val,
    )
