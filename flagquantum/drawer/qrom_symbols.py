# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROM.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom.py

"""Source QROM labels for native register-score renderers."""

from .register_score import RegisterStep
from .register_score_data import PortSide, SymbolClass


def qrom_wire_symbol(
    step: RegisterStep, register: str | None, side: PortSide | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Resolve a QROM glyph without changing the underlying register name."""
    if step.gate != "qrom":
        raise ValueError("A QROM display step is required")
    if register is None:
        return "Text", {"text": "QROM"}
    if register == "control":
        return "Circle", {}
    if register == "selection":
        return "TextBox", {"text": "In"}
    if register.startswith("selection"):
        index = int(register.removeprefix("selection"))
        return "TextBox", {"text": f"In_{chr(ord('i') + index)}"}
    if register.startswith("target") and register.endswith("_"):
        index = int(register.removeprefix("target").removesuffix("_"))
        return "TextBox", {"text": f"QROM_{chr(ord('a') + index)}"}
    raise ValueError(f"Unrecognized QROM register name {register}")
