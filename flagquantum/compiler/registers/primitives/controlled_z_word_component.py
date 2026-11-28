# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Follows Qualtran v0.7.0 chemistry/black_boxes.py ApplyControlledZs.

"""Phase the first system bit through the original split/control/join wrapper."""

from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_builder import ComponentProgramBuilder
from ..component_control import _stored_control_adder, control_component_program
from ..component_graph import _namespace_helpers
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from ..control.multi_control_component import (
    multi_control_z_program,
    positive_controlled_multi_z_program,
)


def positive_controlled_z_word_program(
    width: int, values: tuple[int, ...]
) -> ProgramRecord:
    """Control the original Split/MultiControlZ/Join word phase without flattening.

    The outer scalar ctrl, existing ctrls array and MSB-first system are THRU.
    Split/Join bypass ctrl; the merged-clause predicate owns its restored work.
    """
    record = controlled_z_word_program(width, values)
    controlled = _namespace_helpers(
        positive_controlled_multi_z_program(values), "positive_multi_z"
    )
    middle = record.functions[0].blocks[0].operations[1].callee
    assert middle is not None
    used = {operation.callee for operation in record.functions[0].blocks[0].operations}
    controls = {
        function.name: (
            _stored_control_adder(controlled.functions[0])
            if function.name == middle
            else bookkeeping_scalar_control_adder(function)
        )
        for function in record.functions[1:]
        if function.name in used
    }
    return control_component_program(
        replace(record, functions=(*record.functions, *controlled.functions)), controls
    )


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
