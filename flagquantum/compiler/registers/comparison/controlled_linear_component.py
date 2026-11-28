# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled(LinearDepthGreaterThan).
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Retain controlled carry comparisons up to original directional refusals."""

from dataclasses import replace

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.out_of_place_addition_component import out_of_place_addition_program
from ..bits.bitwise_component import bitwise_word_program
from ..bits.bitwise_component_control import (
    _controlled_cx,
    _controlled_x,
    _toffoli_function,
    controlled_bitwise_word_program,
)
from ..bits.sign_extension_component import sign_extension_program
from ..component_control import (
    ControlAdder,
    component_program_control_adder,
    control_component_program,
    quantum_control_signature,
)
from ..component_definition_match import matches_component_definition
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from ..control.conjunction_program import and_component_program
from ..control.multi_control_component import (
    multi_control_x_program,
    positive_controlled_multi_x_program,
)
from ._component_gates import comparison_bit_functions
from .controlled_component import controlled_greater_program
from .linear_component import linear_greater_program


def controlled_linear_greater_program(
    width: int, *, signed: bool = False, adjoint: bool = False
) -> ProgramRecord:
    """Control the source unsigned-port carry comparator, preserving MSB order.

    Width one has a complete multi-control predicate body. Wider comparators
    retain the original non-THRU Controlled(And) and Controlled(And-adjoint)
    declarations. Those source constructors refuse decomposition; binding must
    fail rather than invent a controlled measurement or claim clean workspace.
    The signed flag preserves the original comparator's overflow behavior.
    """
    record = linear_greater_program(width, signed=signed, adjoint=adjoint)
    if width == 1:
        calls = record.functions[0].blocks[0].operations
        assert len(calls) == 1 and calls[0].callee is not None
        return control_component_program(
            record,
            {
                calls[0].callee: component_program_control_adder(
                    positive_controlled_multi_x_program((1, 0))
                )
            },
        )
    conjunctions = tuple(
        and_component_program((1, 1), name="carry", uncompute=inverse)
        for inverse in (False, True)
    )
    return _control_comparison(record, (), conjunctions)


def nested_controlled_greater_program(
    dtype: RegisterType, *, control_value: int = 1, adjoint: bool = False
) -> ProgramRecord:
    """Preserve an added positive control on the existing controlled comparator.

    Available complements and merged multi-control predicates retain full
    bodies. Only the source-refused directional adder and sign-extension
    controls remain declarations. No numerical or workspace claim covers those
    unresolved declarations; the existing signed-adjoint refusal is retained.
    """
    record = controlled_greater_program(
        dtype, control_value=control_value, adjoint=adjoint
    )
    width = dtype.num_qubits
    assert type(width) is int
    wide = replace(dtype, quantum_bits=width + 1)
    complement_types = (wide, unsigned_register_type(width + 2))
    owners = tuple(
        (
            bitwise_word_program(word, "bitwise_not"),
            controlled_bitwise_word_program(word, "bitwise_not"),
        )
        for word in complement_types
    ) + (
        (
            multi_control_x_program((control_value, 1)),
            positive_controlled_multi_x_program((control_value, 1)),
        ),
    )
    refused = tuple(
        out_of_place_addition_program(width + 1, adjoint=inverse)
        for inverse in (False, True)
    )
    if dtype.encoding == "signed":
        refused += tuple(
            sign_extension_program(dtype, wide, adjoint=inverse)
            for inverse in (False, True)
        )
    return _control_comparison(record, owners, refused)


def _control_comparison(
    record: ProgramRecord,
    owners: tuple[tuple[ProgramRecord, ProgramRecord], ...],
    refused: tuple[ProgramRecord, ...],
) -> ProgramRecord:
    x_gate, cx_gate = comparison_bit_functions()
    adders: dict[str, ControlAdder] = {}
    functions = {function.name: function for function in record.functions}
    for operation in record.functions[0].blocks[0].operations:
        assert operation.callee is not None
        name = operation.callee
        if name in adders:
            continue
        selected = next(
            (
                controlled
                for original, controlled in owners
                if matches_component_definition(record, name, original)
            ),
            None,
        )
        if selected is not None:
            adders[name] = component_program_control_adder(selected)
            continue
        for original, callback in ((x_gate, _controlled_x), (cx_gate, _controlled_cx)):
            if matches_component_definition(record, name, ProgramRecord((original,))):
                adders[name] = callback
                break
        else:
            if any(matches_component_definition(record, name, c) for c in refused):
                registers = functions[name].registers
                assert registers is not None
                signature = quantum_control_signature(
                    Signature(registers), ((quantum_bit_type(), ()),)
                )
                declaration = FunctionRecord(
                    name + "_controlled", (), registers=tuple(signature)
                )
                adders[name] = component_program_control_adder(
                    ProgramRecord((declaration,))
                )
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
