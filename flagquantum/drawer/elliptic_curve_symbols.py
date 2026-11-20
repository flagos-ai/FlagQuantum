# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ECAddR and ECWindowAddR.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add_r.py

"""Fixed-point addition glyphs for the owned register-score callback."""

from __future__ import annotations

from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicInt
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def elliptic_curve_addition_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    x: SymbolicInt,
    y: SymbolicInt,
    windowed_width: SymbolicInt | None = None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain source scalar and row-major control-array glyphs and root labels."""
    if register is None:
        text = (
            "" if windowed_width is None else f"ECWindowAddR(self.n={windowed_width!r})"
        )
        return "Text", {"text": text, "fontsize": 10}
    if register.name == "ctrl":
        return "Circle", {"filled": True}
    if register.name == "x":
        return "TextBox", {"text": f"$+{x}$"}
    if register.name == "y":
        return "TextBox", {"text": f"$+{y}$"}
    raise ValueError(f"Unrecognized register name {register.name}")
