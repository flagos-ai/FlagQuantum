# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotation.py CZPowGate and CRz.

"""Retain exact controlled-rotation calls and shared measured And cleanup."""

from math import pi

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.parameters import Parameter, ParameterExpression
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.conjunction_program import and_component_program
from .global_phase import _finite_angle

RotationAngle = float | Parameter | ParameterExpression


def _z_power_function(name: str, exponent: RotationAngle) -> FunctionRecord:
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


def controlled_z_power_program(exponent: RotationAngle) -> ProgramRecord:
    """Phase two positive controls through a produced and measured And target.

    The two q elements are MSB-first through ports. The And owners allocate
    and erase their temporary target. Binding supplies one clean work qubit.
    """
    compute = and_component_program((1, 1), name="compute")
    erase = and_component_program((1, 1), name="erase", uncompute=True)
    phase = _z_power_function("z_power", exponent)
    builder = ComponentProgramBuilder(
        Signature((Register("q", quantum_bit_type(), (2,)),)),
        (*compute.functions, *erase.functions, phase),
    )
    result = builder.call_dict("compute", {"ctrl": builder.inputs["q"]})
    target = builder.call_dict("z_power", {"q": result["target"]})["q"]
    controls = builder.call_dict("erase", {"ctrl": result["ctrl"], "target": target})[
        "ctrl"
    ]
    return builder.finalize({"q": controls})


def controlled_rz_program(angle: RotationAngle) -> ProgramRecord:
    """Retain the source's two phase powers and two CNOTs in radians.

    ctrl and q are one-bit through ports. The source division order is kept
    for numerical values and owned symbolic expressions. This uses no work.
    """
    half: RotationAngle = (
        ParameterExpression("div", (ParameterExpression("div", (angle, pi)), 2))
        if isinstance(angle, (Parameter, ParameterExpression))
        else (_finite_angle(angle) / pi) / 2
    )
    cnot = bit_gate_function(
        "cnot",
        (Register("ctrl", quantum_bit_type()), Register("target", quantum_bit_type())),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature.build(ctrl=1, q=1),
        (
            _z_power_function("positive", half),
            _z_power_function("negative", -half),
            cnot,
        ),
    )
    ctrl, q = builder.inputs["ctrl"], builder.inputs["q"]
    for phase in ("positive", "negative"):
        q = builder.call_dict(phase, {"q": q})["q"]
        result = builder.call_dict("cnot", {"ctrl": ctrl, "target": q})
        ctrl, q = result["ctrl"], result["target"]
    return builder.finalize({"ctrl": ctrl, "q": q})
