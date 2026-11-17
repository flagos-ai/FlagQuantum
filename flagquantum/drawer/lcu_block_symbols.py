# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SELECT and LCU block encoding wire symbols.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/lcu_block_encoding.py

"""Register-score callbacks for SELECT and LCU block encodings."""

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def select_block_wire_symbol(
    step: RegisterStep, register: Register | None, idx: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Display a block box on each SELECT port and an empty connector label."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    return "TextBox", {"text": "B[H]"}


def lcu_block_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
    *,
    control_value: int | None = None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain the source's literal control-name rule, including its ctrl label."""
    if register is not None and register.name == "control":
        return "Circle", {"filled": bool(control_value)}
    return select_block_wire_symbol(step, register, idx)
