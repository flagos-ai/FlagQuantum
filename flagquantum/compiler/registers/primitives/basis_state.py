# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 X/Z basis state emission.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Prepare a pinned basis vector on one caller-owned clean wire."""

from typing import Literal

from ....core.ir import Instruction
from ..._component_trace.resource_generalization import ZeroCostCall
from ..bits.bitwise import _validate_wires

Basis = Literal["x", "z"]


def _validate_basis_vector(basis: Basis, bit: int) -> None:
    if basis not in ("x", "z") or type(bit) is not int or bit not in (0, 1):
        raise ValueError("Basis vector requires X or Z and a binary bit")


def basis_state_ir(basis: Basis, bit: int, target: int) -> tuple[Instruction, ...]:
    """Prepare zero/one (Z) or plus/minus (X) from a clean output wire."""
    _validate_basis_vector(basis, bit)
    _validate_wires((target,))
    flips = (Instruction("x", (target,)),) if bit else ()
    return (*flips, Instruction("h", (target,))) if basis == "x" else flips


def basis_state_cost_key(basis: Basis, bit: int) -> ZeroCostCall:
    """Retain the source's zero QEC cost and distinguish the prepared vectors."""
    _validate_basis_vector(basis, bit)
    return ZeroCostCall("x_state" if basis == "x" else "z_state", (bit,))


def basis_effect_cost_key(basis: Basis, bit: int) -> ZeroCostCall:
    """Retain the source's zero QEC cost for a left-sided basis covector."""
    _validate_basis_vector(basis, bit)
    return ZeroCostCall("x_effect" if basis == "x" else "z_effect", (bit,))
