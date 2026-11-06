# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 IntState and IntEffect.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Bind integer basis vectors and covectors in big-endian wire order."""

from typing import Literal

from ...core.ir import Instruction
from .._component_trace.resource_generalization import CliffordCall
from .._component_trace.symbolic_math import SymbolicInt
from .bitwise import _validate_unsigned_constant, _validate_wires, xor_constant_ir


def int_state_ir(target: tuple[int, ...], value: int) -> tuple[Instruction, ...]:
    """Prepare the unsigned integer on a nonempty caller-owned clean register."""
    return xor_constant_ir(target, value)


def int_effect_bindings(
    target: tuple[int, ...], value: int
) -> dict[int, tuple[Literal["z"], int]]:
    """Bind the source's left-sided Z covectors for exact partial contraction."""
    _validate_wires(target)
    _validate_unsigned_constant(len(target), value)
    return {
        wire: ("z", (value >> position) & 1)
        for position, wire in zip(reversed(range(len(target))), target, strict=True)
    }


def int_vector_cost_key(width: SymbolicInt) -> CliffordCall:
    """Return the source's single ArbitraryClifford call for either side."""
    if isinstance(width, (int, float)) and (type(width) is not int or width < 0):
        raise ValueError("Integer vector cost requires a nonnegative integer width")
    return CliffordCall("arbitrary", width)
