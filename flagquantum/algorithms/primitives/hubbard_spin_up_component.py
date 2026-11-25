# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Store Hubbard spin-up partitioning and inlined coordinate selection."""

from ...compiler.registers.lookup.hubbard_spin_up_component import (
    hubbard_spin_up_program as _program,
)
from ...core.ir.program import ProgramRecord
from .hubbard_spin_up import HubbardSpinUpZ


def hubbard_spin_up_program(
    operation: HubbardSpinUpZ, *, adjoint: bool = False
) -> ProgramRecord:
    """Bind the existing validated operation to the shared typed call owner."""
    if operation.controlled and operation.control_value == 0:
        raise ValueError(
            "Original stored spin-up zero control executes a positive-control body"
        )
    return _program(
        operation.dimension, controlled=operation.controlled, adjoint=adjoint
    )
