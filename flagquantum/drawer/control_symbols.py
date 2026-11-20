# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpec and _ControlledBase.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py
# Multi-control dispatch also follows bloqs/mcmt/multi_control_pauli.py.

"""Present grouped control activation values using native register metadata."""

import numpy as np

from ..core.ir.program.registers import Register, RegisterSide
from .register_score import RegisterStep
from .register_score_data import (
    RegisterWireSymbolSupplier,
    SymbolClass,
    adjoint_wire_symbol,
)


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


def control_predicate_wire_symbol(
    step: "RegisterStep",
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    values: tuple[tuple[int, ...] | None, ...],
    uncompute: bool = False,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Present grouped equality controls and directional predicate outputs."""
    if uncompute:
        return adjoint_wire_symbol(
            *control_predicate_wire_symbol(
                step,
                None if register is None else register.adjoint(),
                index,
                values=values,
            )
        )
    if register is None:
        return "Text", {"text": "Ctrl", "fontsize": 10}
    if register.name.startswith("ctrl_"):
        if index is None:
            raise ValueError("Control presentation requires an element index")
        group = int(register.name[5:])
        return control_register_wire_symbol(register, index, values[group])
    text = "C" if register.name == "target" else register.name
    if register.name != "target" and index:
        text += "[" + ", ".join(map(str, index)) + "]"
    glyphs: dict[RegisterSide, SymbolClass] = {
        RegisterSide.LEFT: "RarrowTextBox",
        RegisterSide.RIGHT: "LarrowTextBox",
        RegisterSide.THRU: "TextBox",
    }
    return glyphs[register.side], {"text": text}


def multi_control_wire_symbol(
    step: "RegisterStep",
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    values: tuple[int, ...] | None,
    target_register: Register,
    target_symbol: RegisterWireSymbolSupplier,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Dispatch a single-target child glyph and ordered control polarities.

    The caller supplies the owned child's register and glyph callback. None
    activation denotes a symbolic control array, whose individual glyphs cannot
    be resolved. The step name retains the original whole-operation title,
    rendered as the native connector Text glyph. The source uses a TextBox title.
    """
    if register is None:
        return "Text", {"text": step.name, "fontsize": 10}
    if register.name == "target":
        return target_symbol(step, target_register, ())
    if values is None:
        raise ValueError("Multi-control wire polarities are symbolic")
    if index is None:
        raise ValueError("Multi-control presentation requires an element index")
    return control_register_wire_symbol(register, index, values)
