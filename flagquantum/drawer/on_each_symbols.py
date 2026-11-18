# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OnEach.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py

"""Render a repeated single-bit child through native score callbacks."""

from ..core.ir.program.registers import Register, RegisterSide
from ..core.value_math import SymbolicInt
from .register_score import RegisterStep
from .register_score_data import RegisterWireSymbolSupplier, SymbolClass


def on_each_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    width: SymbolicInt,
    child_symbol: RegisterWireSymbolSupplier,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Decorate child text or use the source's default register presentation.

    The display step name is the whole operation's title, including its tensor
    suffix. The child callback receives the original register and index. Its
    attributes remain caller-owned and are never mutated.
    """
    symbol, attributes = child_symbol(step, register, index)
    if symbol in {"Text", "TextBox"}:
        text = attributes["text"]
        if not isinstance(text, str):
            raise TypeError("text wire symbols require string text")
        result: dict[str, str | int | bool] = {
            "text": "" if symbol == "Text" and not text else f"{text}⨂{width}"
        }
        if symbol == "Text":
            result["fontsize"] = 10
        return symbol, result
    if register is None:
        title = step.name if len(step.name) <= 10 else step.name[:8] + ".."
        return "Text", {"text": title, "fontsize": 10}
    label = register.name
    if index:
        label += "[" + ", ".join(map(str, index)) + "]"
    glyphs: dict[RegisterSide, SymbolClass] = {
        RegisterSide.LEFT: "RarrowTextBox",
        RegisterSide.RIGHT: "LarrowTextBox",
        RegisterSide.THRU: "TextBox",
    }
    return glyphs[register.side], {"text": label}
