# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 And/MultiAnd presentation protocols.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py

"""Conjunction glyphs preserve individual symbolic control metadata."""

from __future__ import annotations

from ..core.ir.program.registers import Register, RegisterSide
from ..core.value_math import SymbolicInt
from ..core.value_parameters import is_symbolic_value
from .register_score import RegisterStep
from .register_score_data import SymbolClass

_DIRECTIONAL_BOXES: dict[RegisterSide, SymbolClass] = {
    RegisterSide.THRU: "TextBox",
    RegisterSide.LEFT: "RarrowTextBox",
    RegisterSide.RIGHT: "LarrowTextBox",
}


def and_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    values: tuple[SymbolicInt, SymbolicInt] = (1, 1),
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain atomic And's truth-value circle rule and directed output box."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name == "target":
        return _DIRECTIONAL_BOXES[register.side], {"text": "∧"}
    if index is None:
        raise ValueError("And control presentation requires an element index")
    (control,) = index
    return "Circle", {"filled": bool(values[0] if control == 0 else values[1])}


def multi_and_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    values: tuple[SymbolicInt, ...] | SymbolicInt,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain MultiAnd's equality-to-one circle rule and named junk boxes."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name == "ctrl":
        if isinstance(values, int):
            values = (1,) * values
        if not isinstance(values, tuple):
            raise ValueError(f"Control count {values} has no concrete control values")
        if index is None:
            raise ValueError("MultiAnd control presentation requires an element index")
        return "Circle", {"filled": bool(values[index[0]] == 1)}
    text = "∧" if register.name == "target" else register.name
    if register.name != "target" and index:
        text += "[" + ", ".join(map(str, index)) + "]"
    return _DIRECTIONAL_BOXES[register.side], {"text": text}


def and_diagram_symbols(
    values: tuple[SymbolicInt, SymbolicInt] = (1, 1), *, uncompute: bool = False
) -> tuple[str, str, str]:
    """Use symbolic labels for both controls when either is symbolic."""
    target = "And†" if uncompute else "And"
    if is_symbolic_value(*values):
        return str(values[0]), str(values[1]), target
    controls = ("(0)", "@")
    return controls[int(values[0])], controls[int(values[1])], target
