# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 x_basis.py and z_basis.py wire symbols.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates

"""Native score labels for basis vectors and Pauli atoms."""

from typing import Literal

from ..core.ir.program.registers import Register
from .partition_symbols import partition_wire_symbol
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def basis_vector_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    label: str,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Use the vector value and the existing directional register glyph."""
    symbol, attributes = partition_wire_symbol(step, register, ())
    return symbol, attributes if register is None else {"text": label}


def pauli_basis_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    index: tuple[int, ...] | None,
    *,
    gate: Literal["x", "z", "cz"],
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Preserve source Pauli titles, labels and CZ name validation."""
    if register is None:
        return "Text", {"text": "X" if gate == "x" else "", "fontsize": 10}
    if gate == "x":
        return "ModPlus", {}
    if gate == "z":
        return "TextBox", {"text": "Z"}
    if register.name in {"q1", "q2"}:
        return "Circle", {"filled": True}
    raise ValueError(f"Unknown wire symbol register name: {register.name}")
