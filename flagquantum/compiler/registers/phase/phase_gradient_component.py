# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/phase_gradient.py.

"""Retain gradient powers and preparation as owned typed component calls."""

from dataclasses import replace

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.parameters import Parameter, ParameterExpression
from ..bits.on_each import on_each_word_program
from ..component_adjoint import adjoint_component_program
from ..component_builder import ComponentProgramBuilder
from ..component_definition_match import matches_component_definition
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from ..component_rebindings import adjoint_word_rebinding
from ..primitives.controlled_rotation_component import controlled_z_power_program
from ..primitives.pauli_rotation import RotationAngle, _rotation_angle
from ..primitives.pauli_rotation_component import z_power_function
from .phase_gradient import (
    source_phase_gradient_state_signature,
    source_phase_gradient_unitary_signature,
)


def phase_gradient_unitary_program(
    width: int,
    exponent: RotationAngle = 1,
    *,
    controlled: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Apply source-ordered Z powers to an MSB-first fractional through word.

    Uncontrolled powers need no workspace. Controlled powers own and erase one
    shared And target per call. Source Z powers remain phase primitives. A
    one-bit source word is reused without an extra Split/Join pair.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Gradient component width must be a positive integer")
    _rotation_angle(exponent)
    if type(controlled) is not bool or type(adjoint) is not bool:
        raise ValueError("Gradient controlled mode and adjoint must be boolean")
    powers = tuple(
        (
            ParameterExpression("div", (exponent, 2**i))
            if isinstance(exponent, (Parameter, ParameterExpression))
            else exponent / 2**i
        )
        for i in range(width)
    )
    phases = tuple(
        z_power_function(f"power_{i}", power) for i, power in enumerate(powers)
    )
    builder = ComponentProgramBuilder(
        source_phase_gradient_unitary_signature(width, controlled=controlled),
        () if controlled else phases,
    )
    word = builder.inputs["phase_grad"]
    assert isinstance(word, RegisterBinding)
    bits = [word] if width == 1 else builder.split(word)
    control = builder.inputs.get("ctrl")
    for i, phase in enumerate(phases):
        if controlled:
            returned = builder.call_program(
                controlled_z_power_program(powers[i]),
                {"q": np.array([control, bits[i]], dtype=object)},
            )["q"]
            control, bits[i] = np.asarray(returned)
        else:
            value = builder.call_dict(phase.name, {"q": bits[i]})["q"]
            assert isinstance(value, RegisterBinding)
            bits[i] = value
    outputs = {
        "phase_grad": (
            bits[0]
            if width == 1
            else builder.join(bits, fixed_point_register_type(width, width))
        )
    }
    if controlled:
        assert isinstance(control, RegisterBinding)
        outputs["ctrl"] = control
    record = builder.finalize(outputs)
    if not adjoint:
        return record
    definitions = list(record.functions)
    adjoints: dict[str, str] = {}
    callees = {op.callee for op in record.functions[0].blocks[0].operations}
    for position, function in enumerate(record.functions[1:]):
        if function.name not in callees:
            continue
        ports = function.registers or ()
        if tuple(port.name for port in ports) == ("reg", "reg"):
            inverse = adjoint_word_rebinding(
                function, f"gradient_inverse_view_{position}"
            )
            definitions.append(inverse)
            adjoints[function.name] = inverse.name
        elif not controlled:
            i = int(function.name.removeprefix("power_"))
            inverse = z_power_function(f"gradient_inverse_power_{i}", -powers[i])
            definitions.append(inverse)
            adjoints[function.name] = inverse.name
        else:
            for i in range(width):
                if matches_component_definition(
                    record, function.name, controlled_z_power_program(powers[i])
                ):
                    inverse_record = _namespace_helpers(
                        controlled_z_power_program(-powers[i]),
                        f"gradient_inverse_cz_{i}",
                    )
                    definitions.extend(inverse_record.functions)
                    adjoints[function.name] = inverse_record.functions[0].name
                    break
    return adjoint_component_program(
        replace(record, functions=tuple(definitions)), adjoints
    )


def phase_gradient_state_program(
    width: int, exponent: RotationAngle = -1
) -> ProgramRecord:
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
