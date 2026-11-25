# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Retain Hubbard Majorana's inlined shared-register decomposition."""

from ...compiler.registers.lookup.hubbard_majorana_component import (
    hubbard_majorana_program as _program,
)
from ...core.ir.program import ProgramRecord
from .hubbard_majorana import HubbardMajorana


def hubbard_majorana_program(
    operation: HubbardMajorana, *, adjoint: bool = False
) -> ProgramRecord:
    """Bind the existing validated operation to the shared typed call owner."""
    if operation.controlled and operation.control_value == 0:
        raise ValueError(
            "Original Hubbard Majorana stored decomposition cannot bind zero control"
        )
    return _program(
        operation.dimension,
        operation.pauli,
        controlled=operation.controlled,
        adjoint=adjoint,
    )
