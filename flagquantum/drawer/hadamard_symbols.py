# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Hadamard and CHadamard.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/hadamard.py

"""Hadamard atom labels for the typed register-score callback."""

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def hadamard_wire_symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Return the source's empty title and constant H register label."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    return "TextBox", {"text": "H"}


def controlled_hadamard_wire_symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Return a filled control and H target, rejecting unknown registers."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    if register.name == "ctrl":
        return "Circle", {"filled": True}
    if register.name == "target":
        return "TextBox", {"text": "H"}
    raise ValueError(f"Unknown register {register}")
