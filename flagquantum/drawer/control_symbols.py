# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpec and _ControlledBase.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Present grouped control activation values using native register metadata."""

import numpy as np

from ..core.ir.program.registers import Register
from .register_score_data import SymbolClass


def control_register_wire_symbol(
    register: Register, index: tuple[int, ...], values: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Draw a bit polarity or word value, with None for an unknown activation.

    Values use row-major register order. Their encoding domain belongs to the
    control emitter. Presentation retains source metadata such as a signed
    word's out-of-range integer without asserting that it can execute.
    """
    if values is None:
        return "TextBox", {"text": "ctrl"}
    value = np.asarray(values).reshape(register.shape)[index]
    if register.bitsize == 1:
        return "Circle", {"filled": bool(value == 1)}
    return "TextBox", {"text": str(value)}


def controlled_wire_title(
    symbol: SymbolClass, attributes: dict[str, str | int | bool]
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Wrap an existing Text title while preserving the source empty-title rule."""
    if symbol != "Text":
        raise ValueError("a controlled child must provide a Text title")
    text = attributes["text"]
    return "Text", {"text": f"C[{text}]" if text != "" else "", "fontsize": 10}
