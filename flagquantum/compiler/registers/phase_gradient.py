# Adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC (phase_gradient.py)
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs
# Changes: emit native IR.

"""Phase-gradient state in FlagQuantum IR."""

from math import pi

from ...core.ir import Instruction


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
