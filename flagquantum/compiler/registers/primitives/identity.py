# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Identity and its quantum-control binding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/identity.py

"""Native binding for a register-wide atomic identity."""

from ....core.ir import Instruction
from ..bits.bitwise import on_each_ir


def register_identity_ir(
    target: tuple[int, ...],
    *,
    controls: tuple[tuple[int, ...], ...] = (),
) -> tuple[Instruction, ...]:
    """Emit explicit identity leaves on the packed controls and target.

    The source packs control groups before the target. Quantum activation
    values have no effect on the identity and need no predicate or work.
    """
    control_wires = tuple(wire for group in controls for wire in group)
    return on_each_ir(Instruction("i", (0,)), (*control_wires, *target))
