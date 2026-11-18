# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotation atom wire symbols.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Pauli power and radian labels for typed register scores."""

from typing import Literal

from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicReal
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def rotation_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    axis: Literal["x", "y", "z"],
    parameter: SymbolicReal,
    power: bool = False,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain source parameter strings, exponent labels and empty root title."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    label = f"{axis.upper()}^{parameter}" if power else f"R{axis}({parameter})"
    return "TextBox", {"text": label}
