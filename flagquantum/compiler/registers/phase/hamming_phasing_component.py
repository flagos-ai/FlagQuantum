# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/hamming_weight_phasing.py.

"""Compose count, weighted phase and exact count erasure as stored calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.hamming_weight import hamming_weight_requirements
from ..bits.hamming_weight_component import hamming_weight_program
from ..component_builder import ComponentProgramBuilder
from ..primitives.pauli_rotation_component import z_power_function
from .qvr_component import qvr_phase_gradient_program


def hamming_phasing_program(
    width: int, exponent: float, epsilon: float, *, via_gradient: bool = False
) -> ProgramRecord:
    """Retain MSB-first input and private count/junk ownership until erasure.

    The caller owns the through gradient when selected. The shared counter owns
    its RIGHT words, the exact adjoint consumes them, and phase adders own their
    clean measured carries. Direct phases follow the original LSB-first order.
    Epsilon controls gradient sizing through the existing QVR owner.
    """
    _, count_width = hamming_weight_requirements(width)
    cost = Register("out", fixed_point_register_type(count_width, 0))
    phase = (
        qvr_phase_gradient_program(cost, exponent / 2, epsilon)
        if via_gradient
        else None
    )
    registers: tuple[Register, ...] = (Register("x", RegisterType("uint", width)),)
    if phase is not None:
        registers += (
            Signature(phase.functions[0].registers or ()).get_left("phase_grad"),
        )
    powers = (
        tuple(
            z_power_function(f"weight_phase_{i}", (2**i) * exponent)
            for i in range(count_width)
        )
        if phase is None
        else ()
    )
    builder = ComponentProgramBuilder(Signature(registers), powers)
    values = builder.call_program(
        hamming_weight_program(width), {"x": builder.inputs["x"]}
    )
    if phase is not None:
        rotated = builder.call_program(
            phase, {"out": values["out"], "phase_grad": builder.inputs["phase_grad"]}
        )
        values["out"] = rotated["out"]
    else:
        word = values["out"]
        assert isinstance(word, RegisterBinding)
        bits = builder.split(word)
        for i, function in enumerate(powers):
            bits[-i - 1] = builder.call_dict(function.name, {"q": bits[-i - 1]})["q"]
        values["out"] = builder.join(bits, RegisterType("uint", count_width))
    erased = builder.call_program(hamming_weight_program(width, adjoint=True), values)
    outputs = {"x": erased["x"]}
    if phase is not None:
        outputs["phase_grad"] = rotated["phase_grad"]
    return builder.finalize(outputs)
