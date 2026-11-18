# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 s_gate.py and t_gate.py wire_symbol methods.

"""Fixed S/T phase labels for owned register scores."""

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def fixed_phase_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Keep the root title empty and display the directional gate on its port."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    return "TextBox", {"text": step.name}
