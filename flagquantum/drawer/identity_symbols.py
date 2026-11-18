# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Identity.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/identity.py

"""Identity labels for the existing typed register-score callback."""

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def identity_wire_symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Return the source's empty title and constant I register label."""
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    return "TextBox", {"text": "I"}
