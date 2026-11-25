# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Follows Qualtran v0.7.0 chemistry/black_boxes.py ApplyControlledZs.

"""Phase the first system bit through the original split/control/join wrapper."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_builder import ComponentProgramBuilder
from ..control.multi_control_component import multi_control_z_program


def controlled_z_word_signature(width: SymbolicInt, controls: int) -> Signature:
    """Declare shaped binary controls followed by one opaque system word."""
    return Signature(
        (
            Register("ctrls", quantum_bit_type(), shape=(controls,)),
            Register("system", opaque_register_type(width)),
        )
    )


def controlled_z_word_program(width: int, values: tuple[int, ...]) -> ProgramRecord:
    """Preserve an MSB-first word and phase its first bit under binary controls.

    The existing multi-control owner supplies predicate work and exact erasure.
    This wrapper introduces no additional workspace or measurements.
    """
    if type(width) is not int or width < 1 or not values:
        raise ValueError("Controlled Z word requires a nonempty system and controls")
    word = opaque_register_type(width)
    builder = ComponentProgramBuilder(controlled_z_word_signature(width, len(values)))
    system = builder.inputs["system"]
    assert isinstance(system, RegisterBinding)
    bits = builder.split(system)
    result = builder.call_program(
        multi_control_z_program(values),
        {"controls": builder.inputs["ctrls"], "target": bits[0]},
    )
    bits[0] = result["target"]
    return builder.finalize(
        {
            "ctrls": result["controls"],
            "system": builder.join(bits, word),
        }
    )
