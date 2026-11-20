# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Subtract and SubtractFrom.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/subtraction.py

"""Supply subtraction labels to the existing typed register-score callback."""

from ..core.ir.program.registers import Register, RegisterSide
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def subtraction_wire_symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain source title, operand labels and directional register glyphs."""
    if step.gate not in {"subtract", "subtract_from"}:
        raise ValueError("A subtraction display step is required")
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name not in {"a", "b"}:
        raise ValueError("Subtraction display ports must be a or b")
    label = (
        "a" if register.name == "a" else "a-b" if step.gate == "subtract" else "b - a"
    )
    glyphs: dict[RegisterSide, SymbolClass] = {
        RegisterSide.LEFT: "RarrowTextBox",
        RegisterSide.RIGHT: "LarrowTextBox",
        RegisterSide.THRU: "TextBox",
    }
    return glyphs[register.side], {"text": label}
