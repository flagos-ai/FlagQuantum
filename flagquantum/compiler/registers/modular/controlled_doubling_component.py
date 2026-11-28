# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled(ModDbl).build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Preserve literal control propagation, including the source layout limitation."""

from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ..arithmetic.constant_addition_component import constant_addition_program
from ..arithmetic.nested_controlled_constant_component import (
    nested_controlled_constant_addition_program,
)
from ..bits.bitwise_component_control import (
    _controlled_cx,
    _controlled_x,
    _toffoli_function,
)
from ..comparison._component_gates import comparison_bit_functions
from ..component_control import (
    ControlAdder,
    component_program_control_adder,
    control_component_program,
)
from ..component_definition_match import matches_component_definition
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from .modular_doubling_component import modular_doubling_program


def controlled_modular_doubling_program(
    dtype: RegisterType, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Thread a positive scalar control through the fifteen original calls.

    The data word is MSB first. Bookkeeping remains unconditional, as in the
    source protocol. Consequently the inactive sector is not generally an
    identity: output rebinding can move data or release dirty work. This is a
    literal source decomposition, not a corrected controlled doubling operator.
    Both arithmetic children retain their exact forward or inverse owners.
    """
    record = modular_doubling_program(dtype, modulus, adjoint=adjoint)
    width = executable_register_width(dtype)
    signed = signed_register_type(width + 2)
    unsigned = unsigned_register_type(width + 1)
    pattern = (-modulus) % (1 << (width + 2))
    templates = (
        (
            constant_addition_program(signed, pattern, adjoint=adjoint),
            constant_addition_program(
                signed, pattern, adjoint=adjoint, control_value=1
            ),
        ),
        (
            constant_addition_program(
                unsigned, modulus, adjoint=adjoint, control_value=1
            ),
            nested_controlled_constant_addition_program(
                unsigned, modulus, adjoint=adjoint
            ),
        ),
    )
    x_gate, cx_gate = comparison_bit_functions()
    gates = {
        x_gate: _controlled_x,
        cx_gate: _controlled_cx,
    }
    functions = {function.name: function for function in record.functions}
    adders: dict[str, ControlAdder] = {}
    for operation in record.functions[0].blocks[0].operations:
        assert operation.callee is not None
        name = operation.callee
        if name in adders:
            continue
        for original, controlled in templates:
            if matches_component_definition(record, name, original):
                adders[name] = component_program_control_adder(controlled)
                break
        else:
            for gate, adder in gates.items():
                if matches_component_definition(record, name, ProgramRecord((gate,))):
                    adders[name] = adder
                    break
            else:
                adders[name] = bookkeeping_scalar_control_adder(functions[name])
    return control_component_program(
        replace(
            record,
            functions=(
                *record.functions,
                replace(cx_gate, name="bitwise_controlled_x"),
                _toffoli_function(),
            ),
        ),
        adders,
    )
