# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled(Adjoint(Add/CAdd)).
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Preserve generic inverse controls instead of substituting forward specializations."""

from dataclasses import replace

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord, ValueRef
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component_control import _controlled_cx, _toffoli_function
from ..component_adjoint import adjoint_component_calls
from ..component_control import (
    ControlAdder,
    component_program_control_adder,
    control_component_program,
    quantum_control_signature,
)
from ..component_definition_match import matches_component_definition
from ..component_gate_function import bit_gate_function
from ..component_rebindings import join_word_function, split_word_function
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from ..control.component_predicate import control_predicate_program
from ..control.conjunction_program import and_component_program
from ..control.control_predicate import ControlRegister
from .addition_component import addition_component_program
from .controlled_addition_component import controlled_addition_program


def controlled_inverse_addition_program(
    a_dtype: RegisterType, b_dtype: RegisterType | None = None
) -> ProgramRecord:
    """Retain literal generic control on Add-adjoint, including refused And leaves."""
    target = a_dtype if b_dtype is None else b_dtype
    a_width, b_width = a_dtype.num_qubits, target.num_qubits
    if type(a_width) is not int or type(b_width) is not int:
        raise ValueError("Controlled inverse addition requires concrete widths")
    record = addition_component_program(
        a_width, b_width, left_dtype=a_dtype, right_dtype=target
    )
    if a_width == 1 and b_width > 1:
        record = _without_single_bit_views(record, a_dtype)
    ctrl = (Register("ctrl", quantum_bit_type(), (2,)),)
    compute, erase = (
        control_predicate_program(
            ctrl, (ControlRegister(1, (1, 1)),), name="add_and", uncompute=inverse
        )
        for inverse in (False, True)
    )
    cx = ProgramRecord(
        (next(function for function in record.functions if function.name == "add_cx"),)
    )
    inverse = adjoint_component_calls(
        record, ((compute, erase), (erase, compute), (cx, cx))
    )
    return _control_inverse_calls(inverse, (compute, erase))


def _without_single_bit_views(
    record: ProgramRecord, dtype: RegisterType
) -> ProgramRecord:
    """Match the source Cirq importer, which keeps a one-wire input scalar."""
    root = record.functions[0]
    block = root.blocks[0]
    aliases: dict[ValueRef, ValueRef] = {}
    operations = []
    templates = (
        ProgramRecord((split_word_function("split", dtype),)),
        ProgramRecord((join_word_function("join", dtype),)),
    )
    for operation in block.operations:
        rewritten = replace(
            operation,
            operands=tuple(aliases.get(value, value) for value in operation.operands),
        )
        if operation.callee is not None and any(
            matches_component_definition(record, operation.callee, template)
            for template in templates
        ):
            assert len(rewritten.operands) == len(operation.results) == 1
            aliases[operation.results[0]] = rewritten.operands[0]
        else:
            operations.append(rewritten)
    assert block.return_values is not None
    block = replace(
        block,
        operations=tuple(operations),
        return_values=tuple(aliases.get(value, value) for value in block.return_values),
    )
    return replace(
        record, functions=(replace(root, blocks=(block,)), *record.functions[1:])
    )


def additional_inverse_controlled_addition_program(
    a_dtype: RegisterType,
    b_dtype: RegisterType | None = None,
    *,
    control_value: int = 1,
) -> ProgramRecord:
    """Control CAdd-adjoint through its generic source body, not the three-call specialization."""
    target = a_dtype if b_dtype is None else b_dtype
    record = controlled_addition_program(
        a_dtype, target, control_value=control_value, adjoint=True
    )
    a_width, b_width = a_dtype.num_qubits, target.num_qubits
    assert type(a_width) is int and type(b_width) is int
    original = addition_component_program(
        a_width, b_width, left_dtype=a_dtype, right_dtype=target, adjoint=True
    )
    controlled = controlled_inverse_addition_program(a_dtype, target)
    conjunctions = tuple(
        and_component_program((control_value, 1), name="carry", uncompute=inverse)
        for inverse in (False, True)
    )
    return _control_inverse_calls(record, conjunctions, (original, controlled))


def _control_inverse_calls(
    record: ProgramRecord,
    refused: tuple[ProgramRecord, ...],
    addition: tuple[ProgramRecord, ProgramRecord] | None = None,
) -> ProgramRecord:
    functions = {function.name: function for function in record.functions}
    adders: dict[str, ControlAdder] = {}
    for operation in record.functions[0].blocks[0].operations:
        assert operation.callee is not None
        name = operation.callee
        if name in adders:
            continue
        function = functions[name]
        if (
            addition is not None
            and function.registers == addition[0].functions[0].registers
            and matches_component_definition(record, name, addition[0])
        ):
            adders[name] = component_program_control_adder(addition[1])
        elif any(
            function.registers == original.functions[0].registers
            and matches_component_definition(record, name, original)
            for original in refused
        ):
            assert function.registers is not None
            signature = quantum_control_signature(
                Signature(function.registers), ((quantum_bit_type(), ()),)
            )
            declaration = FunctionRecord(
                name + "_controlled", (), registers=tuple(signature)
            )
            adders[name] = component_program_control_adder(
                ProgramRecord((declaration,))
            )
        elif (
            function.registers is not None
            and tuple(r.name for r in function.registers) == ("ctrl", "target")
            and matches_component_definition(
                record,
                name,
                ProgramRecord(
                    (
                        bit_gate_function(
                            name,
                            function.registers,
                            CircuitIR(2, (Instruction("cx", (0, 1)),)),
                        ),
                    )
                ),
            )
        ):
            adders[name] = _controlled_cx
        else:
            adders[name] = bookkeeping_scalar_control_adder(function)
    return control_component_program(
        replace(record, functions=(*record.functions, _toffoli_function())), adders
    )
