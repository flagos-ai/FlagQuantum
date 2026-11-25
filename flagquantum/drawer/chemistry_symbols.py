# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ApplyControlledZs.wire_symbol.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/black_boxes.py

"""Typed register-score glyphs for chemistry operations."""

from typing import Literal

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def number_phase_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
    *,
    values: tuple[int, ...],
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Draw each declared control's polarity and the system's phase box."""
    if register is None:
        return "Text", {"text": "C" * len(values) + "Z", "fontsize": 10}
    if register.name == "system":
        return "TextBox", {"text": "Z"}
    (control,) = idx or ()
    return "Circle", {"filled": bool(values[control])}


def inverse_sqrt_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
    *,
    newton: bool,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Preserve the locked grid polynomial and Newton labels."""
    if register is None:
        return "Text", {
            "text": "y=x^{-1/2}" if newton else "y~x^{-1/2}",
            "fontsize": 10,
        }
    suffix = "[" + ", ".join(map(str, idx)) + "]" if idx else ""
    return "TextBox", {"text": register.name + suffix}


def grid_energy_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
    *,
    term: Literal["kinetic", "pair", "potential"],
    label: str = "V",
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Preserve the grid Hamiltonian's root and shaped through-port labels.

    Adapted from kinetic.py and potential.py in the pinned Qualtran grid_ham
    package. Kinetic energy has a fixed T label; potential labels are caller-owned.
    """
    if register is None:
        name = "T" if term == "kinetic" else label
        suffix = "_ij" if term == "pair" else ""
        return "Text", {"text": f"U_{name}(dt){suffix}", "fontsize": 10}
    suffix = "[" + ", ".join(map(str, idx)) + "]" if idx else ""
    return "TextBox", {"text": register.name + suffix}


# Copyright 2024 Google LLC
# Source: Qualtran v0.7.0 arithmetic/conversions/contiguous_index.py.
def contiguous_index_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Retain the original THC index labels, including its printed minus sign.

    The source classical polynomial uses nu+1. Its historical drawing uses
    nu-1. This callback reproduces that presentation without changing arithmetic.
    """
    if register is None:
        return "Text", {"text": "", "fontsize": 10}
    label = {"mu": "μ", "nu": "ν"}.get(register.name, "⊕ν(ν-1)/2+μ")
    return "TextBox", {"text": label}


# Source: Qualtran v0.7.0 chemistry/trotter/grid_ham/qvr.py.
def grid_qvr_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Preserve the original grid phase root label and opaque port box."""
    if register is None:
        return "Text", {"text": "e^{i*phi}", "fontsize": 10}
    suffix = "[" + ", ".join(map(str, idx)) + "]" if idx else ""
    return "TextBox", {"text": register.name + suffix}
