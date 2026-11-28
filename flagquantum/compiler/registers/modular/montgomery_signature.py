# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Signature from locked Qualtran v0.7.0 DirtyOutOfPlaceMontgomeryModMul.

"""Retain directional Montgomery words and the complete rounded index history."""

from ....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt


def validate_dirty_montgomery_constructor(
    width: SymbolicInt, window: SymbolicInt, modulus: SymbolicInt
) -> None:
    """Retain the original wrapper's assertion types, messages and check order."""
    if isinstance(modulus, int):
        if not (modulus > 1 and modulus % 2 == 1):
            raise AssertionError
    if isinstance(modulus, int) and isinstance(width, int):
        if not 2 * modulus - 1 < 2**width:
            raise AssertionError(f"bitsize={width} is too small")
    if isinstance(window, int) and isinstance(width, int):
        if width % window != 0:
            raise AssertionError


def montgomery_product_signature(
    width: SymbolicInt,
    window: SymbolicInt,
    modulus: SymbolicInt,
    *,
    adjoint: bool = False,
) -> Signature:
    """Describe source product/history ownership without constructing a body.

    Words are MSB first. History retains window*ceil(width/window); the wrapper
    separately rejects concrete nondivisible windows. Forward outputs transfer
    private work; the inverse consumes that exact history. Unresolved dimensions
    remain logical metadata.
    """
    dtype = montgomery_register_type(width, modulus)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    history_width = ((width + window - 1) // window) * window
    return Signature(
        (
            Register("x", dtype),
            Register("y", dtype),
            Register("target", dtype, side=side),
            Register(
                "qrom_indices",
                montgomery_register_type(history_width, modulus),
                side=side,
            ),
            Register("reduced", quantum_bit_type(), side=side),
        )
    )
