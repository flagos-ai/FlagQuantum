# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/multiplication.py wire_symbol methods.

"""Multiplication labels over existing typed register-score geometry."""

from ..core.ir.program.registers import Register, RegisterSide
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def multiplication_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Keep root formulas, row-major labels and result-side arrows."""
    if register is None:
        labels = {
            "product_accumulation": "result += a*b",
            "product_accumulation_adjoint": "result -= a*b",
            "integer_square": "a^2",
            "sum_of_squares": "SOS",
            "integer_product": "a*b",
            "fixed_point_scaling": "r*i",
            "fixed_point_product": "a*b",
            "fixed_point_square": "a^2",
            "reciprocal": "1/a",
        }
        return "Text", {"text": labels[step.name], "fontsize": 10}
    if register.name == "result" and step.name in {
        "product_accumulation",
        "product_accumulation_adjoint",
    }:
        label = "c-=a*b" if step.name.endswith("_adjoint") else "c+=a*b"
        return "TextBox", {"text": label}
    suffix = "[" + ", ".join(map(str, idx)) + "]" if idx else ""
    glyph: SymbolClass = "TextBox"
    if register.side == RegisterSide.LEFT:
        glyph = "RarrowTextBox"
    elif register.side == RegisterSide.RIGHT:
        glyph = "LarrowTextBox"
    return glyph, {"text": register.name + suffix}
