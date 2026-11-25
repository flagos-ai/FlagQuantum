# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Compose Hubbard SELECT with exact optional control and scalar ownership."""

from ...compiler.registers.control.hubbard_select_component import (
    hubbard_select_program as _program,
)
from ...core.ir.program import ProgramRecord
from .hubbard_select import SelectHubbard


def hubbard_select_program(
    operation: SelectHubbard, *, adjoint: bool = False
) -> ProgramRecord:
    """Bind the existing validated operation to the shared typed call owner."""
    if operation.controlled and operation.control_value == 0:
        raise ValueError("Original stored Hubbard SELECT refuses zero control")
    return _program(
        operation.dimension, controlled=operation.controlled, adjoint=adjoint
    )
