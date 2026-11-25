# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 prepare_hubbard.py.

"""Store the square Hubbard preparation recipe and its exact source adjoint."""

from ...compiler.registers.preparation.hubbard_prepare_component import (
    hubbard_prepare_program as _program,
)
from ...core.ir.program import ProgramRecord
from .hubbard_prepare import PrepareHubbard, _hubbard_preparation_angles


def hubbard_prepare_program(
    prepare: PrepareHubbard, *, adjoint: bool = False
) -> ProgramRecord:
    """Bind validated coefficients to the shared source-ordered call composition."""
    return _program(
        prepare.dimension,
        *_hubbard_preparation_angles(prepare.dimension, prepare.t, prepare.u),
        adjoint=adjoint,
    )
