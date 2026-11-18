# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Toffoli.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/toffoli.py

"""Toffoli presentation for shaped native register scores."""

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def toffoli_wire_symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Preserve the source control array and target glyphs."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name == "ctrl":
        return "Circle", {"filled": True}
    if register.name == "target":
        return "ModPlus", {}
    raise ValueError(f"Unknown wire symbol register name: {register.name}")
