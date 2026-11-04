# Adapted from Qualtran v0.7.0, qualtran/bloqs/arithmetic/controlled_add_or_subtract.py.
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py
# Changes: emit native IR and use FlagQuantum's modular adder.

"""Register-level arithmetic decompositions."""

from ...core.ir import Instruction
from .._arithmetic_kernels import ripple_add_modular


def controlled_add_or_subtract_ir(
    control: int, angle: tuple[int, ...], gradient: tuple[int, ...], carry: int
) -> tuple[Instruction, ...]:
    """Add on zero and subtract on one by conjugating with controlled NOT."""
    complement = tuple(Instruction("cx", (control, wire)) for wire in gradient)
    return (*complement, *ripple_add_modular(angle, gradient, carry), *complement)
