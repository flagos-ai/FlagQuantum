# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata follows locked Qualtran v0.7.0 SelectPauliLCU.nth_operation.

"""Capture signed source Pauli data without its object-array wire codec."""

from typing import Any, cast

from ..compiler.registers.lookup.pauli_select_component import (
    pauli_select_component_pair,
)
from ..core.ir.program import ProgramRecord
from ..errors import SerializationError
from ._pauli_select_rows import sign_of_real_pauli_word


def capture_pauli_select_pair(source: object) -> tuple[ProgramRecord, ProgramRecord]:
    """Replace source payload immediately, retaining its sign-of-real convention."""
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU

    if not isinstance(source, SelectPauliLCU):
        raise SerializationError("Expected an explicit source Pauli SELECT")
    foreign = cast(Any, source)
    words: list[str] = []
    for pauli in foreign.select_unitaries:
        try:
            words.append(
                sign_of_real_pauli_word(pauli.pauli_mask, complex(pauli.coefficient))
            )
        except ValueError as error:
            raise SerializationError(
                "Source sign-of-real Pauli leaf is not unitary"
            ) from error
    return pauli_select_component_pair(
        foreign.selection_bitsize,
        foreign.target_bitsize,
        tuple(words),
        control_value=foreign.control_val,
    )
