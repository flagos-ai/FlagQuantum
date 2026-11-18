# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 arithmetic/bitwise.py wire_symbol methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Bitwise word labels for the existing native register score."""

from typing import Literal

from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicInt
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def bitwise_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    operation: Literal["xor", "xor_constant", "bitwise_not"],
    constant: SymbolicInt = 0,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain the original whole-word labels, including unknown port names."""
    if register is None:
        return "TextBox", {"text": ""}
    if operation == "xor":
        label = "x" if register.name == "x" else "x⊕y"
    elif operation == "xor_constant":
        label = f"⊕{constant}"
    else:
        label = "~x"
    return "TextBox", {"text": label}
