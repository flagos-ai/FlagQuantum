# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 swap wire symbols and CSwap circuit diagram.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Atomic Fredkin and word-swap presentation using native register metadata."""

from operator import index as integer_index
from typing import SupportsIndex

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def fredkin_wire_symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Return the source atomic control or unqualified exchange box."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name == "ctrl":
        return "Circle", {"filled": True}
    return "TextBox", {"text": "×"}


def swap_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    controlled: bool = False,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Label whole x/y words, retaining the source controlled fallback."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name in {"x", "y"}:
        return "TextBox", {"text": f"×({register.name})"}
    if controlled:
        return "Circle", {"filled": True}
    raise ValueError(f"Bad register name {register.name}")


def controlled_swap_diagram_symbols(
    width: SupportsIndex, *, use_unicode: bool = True
) -> tuple[str, ...]:
    """Retain the ordered control/x/y labels for any concrete index width."""
    count = integer_index(width)
    left, right = ("×(x)", "×(y)") if use_unicode else ("swap_x", "swap_y")
    return ("@",) + (left,) * count + (right,) * count
