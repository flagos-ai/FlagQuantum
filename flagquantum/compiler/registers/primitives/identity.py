# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Identity and its quantum-control binding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/identity.py

"""Native binding for a register-wide atomic identity."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..bits.bitwise import on_each_ir
from ..bits.on_each import on_each_word_program
from ..component_gate_function import bit_gate_function


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


def register_identity_program(width: int) -> ProgramRecord:
    """Execute an atomic word Identity through native bit leaves.

    The source Identity has no composite recipe. Split, per-bit identities and
    Join are native execution below that boundary, preserving the opaque word.
    """

    if type(width) is not int or width < 1:
        raise ValueError("Executable word Identity requires a positive integer width")
    identity = bit_gate_function(
        "identity_bit",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, register_identity_ir((0,))),
    )
    return on_each_word_program(width, identity)
