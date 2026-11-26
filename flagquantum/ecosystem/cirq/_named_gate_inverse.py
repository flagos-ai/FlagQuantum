# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Inverse wrapper follows Qualtran v0.7.0 cirq_gate_to_bloq and Adjoint.

"""Derive inverse wrapper calls from explicit exact native component pairs."""

from collections.abc import Callable, Hashable, Mapping
from typing import Any

from ...compiler.formats.legacy_word_factories import _rename_program
from ...compiler.registers.component_adjoint import adjoint_component_calls
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.signatures import Signature
from ._register_import import import_cirq_register_gate


def named_inverse_program(
    original: Any,
    declared: Mapping[Hashable, FunctionRecord],
    pairs: Callable[[], tuple[tuple[ProgramRecord, ProgramRecord], ...]],
    name: Callable[[], str],
    *,
    signature: Signature | None = None,
) -> ProgramRecord | None:
    """Keep source call reversal above measured child specializations.

    Unknown primitive children fail in the shared exact-adjoint owner. SDK
    matrix inversion is not substituted for a missing component pair. An
    explicit signature retains the caller's original grouping of SDK operands.
    """
    program = import_cirq_register_gate(
        original, signature=signature, gate_definitions=declared
    )
    if program is None:
        return None
    return _rename_program(adjoint_component_calls(program, pairs()), name())
