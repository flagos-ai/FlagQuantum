# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 addition.py drawing methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py
# Controlled ports from arithmetic/controlled_addition.py at the same commit.

"""Addition presentation supplied to the existing register-score callbacks."""

from .register_score import RegisterStep
from .register_score_data import PortSide, SymbolClass


def addition_wire_symbol(
    step: RegisterStep, register: str | None, side: PortSide | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain Add's output label and the out-of-place sum connector.

    The caller supplies an addition display step with gate ``add``,
    ``out_of_place_add``, ``constant_add`` or ``controlled_add``.
    Names belong to presentation.
    This callback neither resolves resource identities nor changes execution.
    """
    if step.gate not in {"add", "out_of_place_add", "constant_add", "controlled_add"}:
        raise ValueError("An addition display step is required")
    if register is None:
        title = {"add": "", "out_of_place_add": "c=a+b"}.get(step.gate, step.name)
        return "Text", {
            "text": title if len(title) <= 10 else title[:8] + "..",
            "fontsize": 10,
        }
    label = register
    if step.gate in {"add", "controlled_add"}:
        ports = {"ctrl", "a", "b"} if step.gate == "controlled_add" else {"a", "b"}
        if register not in ports:
            message = (
                "Addition display port is not declared"
                if step.gate == "controlled_add"
                else "Add display ports must be a or b"
            )
            raise ValueError(message)
        label = "a+b" if register == "b" else register
    if step.gate == "out_of_place_add" and register == "a" and side == "THRU":
        label = step.name
    glyphs: dict[PortSide | None, SymbolClass] = {
        "LEFT": "RarrowTextBox",
        "RIGHT": "LarrowTextBox",
        "THRU": "TextBox",
        None: "TextBox",
    }
    glyph = glyphs[side]
    return glyph, {"text": label}


def addition_circuit_labels(left_width: int, right_width: int) -> tuple[str, ...]:
    """Return the frozen bit-level Add labels in big-endian port order."""
    return ("In(x)",) * int(left_width) + ("In(y)/Out(x+y)",) * int(right_width)
