# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Partition, Split and Join wire symbols.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/split.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/join.py

"""Register-score callbacks for partition and word rebinding operations."""

from ..core.ir.program.registers import Register, RegisterSide
from .register_score import RegisterStep
from .register_score_data import SymbolClass

_DIRECTIONAL_GLYPHS: dict[RegisterSide, SymbolClass] = {
    RegisterSide.LEFT: "RarrowTextBox",
    RegisterSide.RIGHT: "LarrowTextBox",
    RegisterSide.THRU: "TextBox",
}


def partition_wire_symbol(
    step: RegisterStep, register: Register | None, idx: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Label typed ports, including row-major array elements, in either direction."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    label = register.name
    if idx:
        indices = ", ".join(str(index) for index in idx)
        label = f"{label}[{indices}]"
    return _DIRECTIONAL_GLYPHS[register.side], {"text": label}


def split_join_wire_symbol(
    step: RegisterStep, register: Register | None, idx: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Display a blank word glyph or an index-only bit glyph in either direction."""
    if register is None:
        return partition_wire_symbol(step, register, idx)
    label = " "
    if register.shape_symbolic:
        indices = ", ".join(str(index) for index in (idx or ()))
        label = f"[{indices}]"
    return _DIRECTIONAL_GLYPHS[register.side], {"text": label}
