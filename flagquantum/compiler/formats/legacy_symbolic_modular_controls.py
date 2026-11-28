# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal control dependencies from Qualtran v0.7.0 mod_arithmetic.

"""Compose symbolic-modulus parents around actual unresolved AddK leaves."""

from dataclasses import replace

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import RegisterType
from ...core.ir.program.signatures import Signature
from ..registers.bits.bitwise_component import bitwise_word_program
from ..registers.bits.bitwise_component_control import (
    _controlled_cx,
    _toffoli_function,
    controlled_bitwise_word_program,
)
from ..registers.comparison._component_gates import comparison_bit_functions
from ..registers.component_control import (
    ControlAdder,
    component_program_control_adder,
    quantum_control_signature,
)
from ..registers.modular.additional_control_component import (
    _control_calls,
    _modular_addition_control_dependencies,
)
from ..registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_component_program,
)
from ..registers.modular.modular_subtraction_component import (
    modular_subtraction_component_program,
)
from .legacy_constant_control_factories import unresolved_constant_addition_program
from .legacy_modular_addition_factories import _symbolic_addition
from .legacy_modular_multiplication_factories import symbolic_modular_doubling_program


def _extra_control(record: ProgramRecord) -> ProgramRecord:
    root = record.functions[0]
    if root.blocks or root.registers is None:
        raise ValueError("A symbolic AddK control requires an explicit declaration")
    return ProgramRecord(
        (
            FunctionRecord(
                root.name + "_controlled",
                (),
                registers=tuple(
                    quantum_control_signature(
                        Signature(root.registers), ((quantum_bit_type(), ()),)
                    )
                ),
            ),
        )
    )


def _constant_controls(
    record: ProgramRecord, constants: tuple[ProgramRecord, ...]
) -> dict[str, ControlAdder]:
    """Bind explicit roles in this closed constructor, rejecting ambiguity.

    The caller supplies each actual AddK expression and direction. The original
    constructor context owns these semantics; ABI equality is not a substitute
    for an unknown child's provenance. Distinct same-ABI roles need named binding.
    """
    used = {op.callee for op in record.functions[0].blocks[0].operations}
    declarations = {
        f.name: f for f in record.functions if f.name in used and not f.blocks
    }
    result = {}
    for constant in constants:
        root = constant.functions[0]
        candidates = [
            name
            for name, function in declarations.items()
            if function.registers == root.registers
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous symbolic constant role in closed modular constructor"
            )
        result[candidates[0]] = component_program_control_adder(
            _extra_control(constant)
        )
    if set(result) != set(declarations):
        raise ValueError("Missing explicit symbolic modular constant provenance")
    return result


def symbolic_additional_modular_addition_program(
    dtype: RegisterType,
    modulus: sympy.Expr,
    *,
    control_value: int | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Preserve 17/14 source calls and exact generic inverse control dispatch."""
    record = _symbolic_addition(
        dtype,
        modulus,
        controlled=control_value is not None,
        control_value=1 if control_value is None else control_value,
        adjoint=adjoint,
    )
    word, wide, addition, adder, comparison, controlled_compare = (
        _modular_addition_control_dependencies(
            dtype, control_value=control_value, adjoint=adjoint
        )
    )
    constants = (
        unresolved_constant_addition_program(
            wide, -modulus, name="subtract_modulus", adjoint=adjoint
        ),
        unresolved_constant_addition_program(
            word, modulus, name="restore_modulus", controlled=True, adjoint=adjoint
        ),
    )
    return _control_calls(
        record,
        (
            (addition, adder),
            (comparison, component_program_control_adder(controlled_compare)),
        ),
        declarations=_constant_controls(record, constants),
    )


def symbolic_modular_subtraction_program(
    dtype: RegisterType,
    modulus: sympy.Expr,
    *,
    control_value: int | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Keep the exact five-call shell, including distinct load/unload directions."""
    load = unresolved_constant_addition_program(
        dtype, modulus + 1, name="load_complement"
    )
    unload = unresolved_constant_addition_program(
        dtype, modulus + 1, name="unload_complement", adjoint=True
    )
    addition = _symbolic_addition(
        (
            dtype
            if control_value is not None
            else montgomery_register_type(dtype.num_qubits, modulus)
        ),
        modulus,
        controlled=control_value is not None,
        control_value=1 if control_value is None else control_value,
        adjoint=adjoint,
    )
    # Reversal exchanges the two inverse constant roles; their stored positions
    # in this complement shell remain load then unload, with the addition reversed.
    return (
        modular_subtraction_component_program(dtype, load, unload, addition)
        if control_value is None
        else controlled_modular_subtraction_component_program(
            dtype, load, unload, addition
        )
    )


def symbolic_additional_modular_subtraction_program(
    dtype: RegisterType,
    modulus: sympy.Expr,
    *,
    control_value: int | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Control each explicit source shell role without equating ABI-only leaves."""
    record = symbolic_modular_subtraction_program(
        dtype, modulus, control_value=control_value, adjoint=adjoint
    )
    constants = (
        unresolved_constant_addition_program(
            dtype, modulus + 1, name="load_complement"
        ),
        unresolved_constant_addition_program(
            dtype, modulus + 1, name="unload_complement", adjoint=True
        ),
    )
    functions = {f.name: f for f in record.functions}
    declarations = {}
    for constant in constants:
        root = constant.functions[0]
        # Both leaves have identical ABI and no body. Bind their explicit names
        # from the shell constructor instead of using structural equivalence.
        assert functions[root.name] == root
        declarations[root.name] = component_program_control_adder(
            _extra_control(constant)
        )
    addition_dtype = (
        dtype
        if control_value is not None
        else montgomery_register_type(dtype.num_qubits, modulus)
    )
    addition = _symbolic_addition(
        addition_dtype,
        modulus,
        controlled=control_value is not None,
        control_value=1 if control_value is None else control_value,
        adjoint=adjoint,
    )
    return _control_calls(
        record,
        (
            (
                bitwise_word_program(dtype, "bitwise_not"),
                component_program_control_adder(
                    controlled_bitwise_word_program(dtype, "bitwise_not")
                ),
            ),
            (
                addition,
                component_program_control_adder(
                    symbolic_additional_modular_addition_program(
                        addition_dtype,
                        modulus,
                        control_value=control_value,
                        adjoint=adjoint,
                    )
                ),
            ),
        ),
        declarations=declarations,
    )


def symbolic_controlled_modular_doubling_program(
    dtype: RegisterType,
    modulus: sympy.Expr,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain all fifteen original calls and the original output/work exchange."""
    width = dtype.num_qubits
    assert type(width) is int
    record = symbolic_modular_doubling_program(dtype, modulus, adjoint=adjoint)
    constants = (
        unresolved_constant_addition_program(
            signed_register_type(width + 2),
            -modulus,
            name="subtract_modulus",
            adjoint=adjoint,
        ),
        unresolved_constant_addition_program(
            unsigned_register_type(width + 1),
            modulus,
            name="restore_modulus",
            controlled=True,
            adjoint=adjoint,
        ),
    )
    _, cx = comparison_bit_functions()
    record = replace(record, functions=(*record.functions, _toffoli_function()))
    return _control_calls(
        record,
        ((ProgramRecord((cx,)), _controlled_cx),),
        declarations=_constant_controls(record, constants),
    )
