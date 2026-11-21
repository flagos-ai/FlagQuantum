# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 basic_gates/rotation.py.

"""Typed single-bit phase powers shared by controlled and gradient graphs."""

from math import pi

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.parameters import Parameter, ParameterExpression
from ..component_gate_function import bit_gate_function
from .global_phase import _finite_angle

RotationAngle = float | Parameter | ParameterExpression


def z_power_function(name: str, exponent: RotationAngle) -> FunctionRecord:
    angle = (
        exponent * pi
        if isinstance(exponent, (Parameter, ParameterExpression))
        else _finite_angle(exponent, pi)
    )
    return bit_gate_function(
        name,
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("phase", (0,), params={"theta": angle}),)),
    )
