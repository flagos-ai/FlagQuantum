# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 IntState and IntEffect.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Bind integer basis vectors and covectors in big-endian wire order."""

from typing import Literal

from ....core.ir import Instruction
from ..._component_trace.resource_generalization import (
    CliffordCall,
    arbitrary_clifford_cost_key,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..bits.bitwise import xor_constant_ir
from .register_bits import split_integer, split_register


def int_state_ir(target: tuple[int, ...], value: int) -> tuple[Instruction, ...]:
    """Prepare the unsigned integer on a nonempty caller-owned clean register."""
    return xor_constant_ir(target, value)


def int_effect_bindings(
    target: tuple[int, ...], value: int
) -> dict[int, tuple[Literal["z"], int]]:
    """Bind the source's left-sided Z covectors for exact partial contraction."""
    if not target:
        raise ValueError("Integer effect requires a nonempty register")
    wires = split_register(target)
    bits = split_integer(value, len(wires))
    return {wire: ("z", bit) for (wire,), bit in zip(wires, bits, strict=True)}


def int_vector_cost_key(width: SymbolicInt) -> CliffordCall:
    """Return the source's single ArbitraryClifford call for either side."""
    return arbitrary_clifford_cost_key(width)
