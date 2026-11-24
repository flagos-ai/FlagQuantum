# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotations/programmable_ancilla_rotation.py.

"""Prepare an owned phase resource through the original three typed calls."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..primitives.pauli_rotation import RotationAngle
from ..primitives.pauli_rotation_component import z_power_function


def programmed_phase_program(exponent: RotationAngle) -> ProgramRecord:
    """Allocate a clean RIGHT QBit and prepare H followed by exact Z power.

    The output belongs to the caller after return. No private work or classical
    slots are needed. Synthesis precision remains in the independent model.
    """
    dtype = quantum_bit_type()
    h = bit_gate_function(
        "hadamard", (Register("q", dtype),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    phase = z_power_function("phase", exponent)
    builder = ComponentProgramBuilder(
        Signature((Register("q", dtype, side=RegisterSide.RIGHT),)), (h, phase)
    )
    allocated = builder.allocate(dtype=dtype)
    prepared = builder.call_dict(h.name, {"q": allocated})["q"]
    output = builder.call_dict(phase.name, {"q": prepared})["q"]
    return builder.finalize({"q": output})
