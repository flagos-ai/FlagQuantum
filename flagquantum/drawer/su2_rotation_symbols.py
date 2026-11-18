# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SU2RotationGate.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/su2_rotation.py

"""SU(2) parameter labels for the typed register-score callback."""

from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicReal
from ..core.value_parameters import is_symbolic_value
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def su2_rotation_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    theta: SymbolicReal,
    phi: SymbolicReal,
    lambd: SymbolicReal,
    global_shift: SymbolicReal = 0,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Keep numeric two-decimal labels and unrounded symbolic parameters."""
    if register is not None:
        return "TextBox", {"text": "SU2"}
    angles = (theta, phi, lambd, global_shift)
    labels = (
        map(str, angles)
        if is_symbolic_value(*angles)
        else (f"{angle:.2f}" for angle in angles)
    )
    return "Text", {"text": "(" + ",".join(labels) + ")", "fontsize": 10}
