# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/phase_gradient.py.

"""Retain gradient powers and preparation as owned typed component calls."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ..bits.on_each import on_each_word_program
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..primitives.global_phase import _finite_angle
from ..primitives.pauli_rotation_component import z_power_function
from .phase_gradient import (
    source_phase_gradient_state_signature,
    source_phase_gradient_unitary_signature,
)


def phase_gradient_unitary_program(width: int, exponent: float = 1) -> ProgramRecord:
    """Apply source-ordered Z powers to an MSB-first fractional through word.

    No workspace is required. Every source Z power remains a phase primitive,
    including angles admitting specialized Clifford or T representations.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Gradient component width must be a positive integer")
    _finite_angle(exponent)
    phases = tuple(
        z_power_function(f"power_{i}", exponent / 2**i) for i in range(width)
    )
    builder = ComponentProgramBuilder(
        source_phase_gradient_unitary_signature(width), phases
    )
    word = builder.inputs["phase_grad"]
    assert isinstance(word, RegisterBinding)
    bits = builder.split(word)
    for i, phase in enumerate(phases):
        bits[i] = builder.call_dict(phase.name, {"q": bits[i]})["q"]
    return builder.finalize(
        {"phase_grad": builder.join(bits, fixed_point_register_type(width, width))}
    )


def phase_gradient_state_program(width: int, exponent: float = -1) -> ProgramRecord:
    """Produce the source's logical output through Allocate, OnEach and gradient.

    Allocation and Hadamards keep the source opaque view. The final gradient
    call returns the fractional type. The allocated output stays live.
    """
    gradient = phase_gradient_unitary_program(width, exponent)
    h = bit_gate_function(
        "hadamard",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("h", (0,)),)),
    )
    builder = ComponentProgramBuilder(source_phase_gradient_state_signature(width))
    word = builder.allocate(width)
    prepared = builder.call_program(on_each_word_program(width, h), {"q": word})["q"]
    result = builder.call_program(gradient, {"phase_grad": prepared})["phase_grad"]
    return builder.finalize({"phase_grad": result})
