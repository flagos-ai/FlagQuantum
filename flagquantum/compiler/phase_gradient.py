# Adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC (phase_gradient.py)
# Copyright 2024 Google LLC (controlled_add_or_subtract.py)
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs
# Changes: emit native IR and use FlagQuantum's modular adder.

"""Phase-gradient state and signed modular addition in FlagQuantum IR."""

from math import pi

from ..core.ir import Instruction
from ._arithmetic_kernels import ripple_add_modular


def phase_gradient_state_ir(
    gradient: tuple[int, ...], *, inverse: bool = False
) -> tuple[Instruction, ...]:
    """Prepare or unprepare the positive-exponent phase-gradient state."""
    hadamards = tuple(Instruction("h", (wire,)) for wire in gradient)
    sign = -1 if inverse else 1
    phases = tuple(
        Instruction("phase", (wire,), params={"theta": sign * pi / (1 << i)})
        for i, wire in enumerate(gradient)
    )
    return (*phases, *hadamards) if inverse else (*hadamards, *phases)


def controlled_add_or_subtract_ir(
    control: int, angle: tuple[int, ...], gradient: tuple[int, ...], carry: int
) -> tuple[Instruction, ...]:
    """Add on zero and subtract on one by conjugating with controlled NOT."""
    complement = tuple(Instruction("cx", (control, wire)) for wire in gradient)
    return (*complement, *ripple_add_modular(angle, gradient, carry), *complement)
