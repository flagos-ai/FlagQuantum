# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 basic_gates/rotation.py.

"""Typed single-bit phase powers shared by controlled and gradient graphs."""

from ....core.ir import CircuitIR
from ....core.ir.program import FunctionRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_gate_function import bit_gate_function
from .pauli_rotation import RotationAngle as RotationAngle
from .pauli_rotation import pauli_power_ir


def z_power_function(name: str, exponent: RotationAngle) -> FunctionRecord:
    return bit_gate_function(
        name,
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, pauli_power_ir("z", 0, exponent)),
    )
