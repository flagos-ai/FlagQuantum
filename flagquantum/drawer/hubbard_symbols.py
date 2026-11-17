# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HubbardMajorannaOperator and HubbardSpinUpZ.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py

"""Register-score symbols for Hubbard SELECT leaves and hopping terms."""

from typing import Literal

from ..core.ir.program.registers import Register
from .register_score import RegisterStep
from .register_score_data import SymbolClass


def hubbard_consumer_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
    *,
    majorana: Literal["X", "Y"] | None,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Draw a Majorana Pauli or the spin-up Z leaf on its declared ports."""
    if register is None:
        return "TextBox", {"text": ""}
    if register.name == "control" or (majorana is None and register.name == "V"):
        return "Circle", {"filled": True}
    if register.name == "target":
        text = (
            r"$Z_{x, y, 1}$"
            if majorana is None
            else rf"$\overrightarrow{{Z}} {majorana}_{{x,y,\sigma}}$"
        )
        return "TextBox", {"text": text}
    return "TextBox", {"text": register.name}


def hubbard_hopping_wire_symbol(
    step: RegisterStep,
    register: Register | None,
    idx: tuple[int, ...] | None,
    *,
    plaquette: bool = False,
    pink: bool = True,
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Preserve hopping.py's plaquette default and both tile color labels.

    HoppingTileHWP inherits HoppingTile.wire_symbol. Its short_name is not
    used by the source register-score renderer.
    """
    if register is None:
        text = "HoppingP.." if plaquette else f"H_h^{'p' if pink else 'g'}"
        return "Text", {"text": text, "fontsize": 10}
    suffix = "[" + ", ".join(map(str, idx)) + "]" if idx else ""
    return "TextBox", {"text": register.name + suffix}
