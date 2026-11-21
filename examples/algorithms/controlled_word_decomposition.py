# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 bitwise.py, on_each.py and controlled.py.

"""Retain controlled complement graphs and qualify their repeated-X descendants."""

from dataclasses import replace
from functools import partial

from examples.algorithms.word_arithmetic_decomposition import word_symbols
from flagquantum.compiler.registers.bits.bitwise_component_control import (
    controlled_bitwise_word_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
from flagquantum.drawer.control_symbols import (
    control_register_wire_symbol,
    controlled_wire_title,
)
from flagquantum.drawer.on_each_symbols import on_each_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    RegisterWireSymbolSupplier,
    SymbolClass,
)


def controlled_complement_program(dtype: RegisterType) -> ProgramRecord:
    """Reuse executable complement control or retain its symbolic repeated-X leaf."""
    if type(dtype.num_qubits) is int:
        return controlled_bitwise_word_program(dtype, "bitwise_not")
    bit = quantum_bit_type()
    child = FunctionRecord(
        "ControlledOnEachX",
        (),
        registers=tuple(Signature.build_from_dtypes(ctrl=bit, q=dtype)),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=bit, x=dtype), (child,)
    )
    result = builder.call_dict(
        child.name, {"ctrl": builder.inputs["ctrl"], "q": builder.inputs["x"]}
    )
    return builder.finalize({"ctrl": result["ctrl"], "x": result["q"]})


def controlled_word_symbols(record: ProgramRecord) -> RegisterWireSymbolSupplier:
    """Compose existing control, repeated-X, CNOT and word partition glyphs."""
    functions = {function.name: function for function in record.functions}
    fallback = word_symbols(record, "Controlled")
    x_symbol = partial(pauli_basis_wire_symbol, gate="x")

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        registers = functions[step.name].registers or ()
        if tuple(r.name for r in registers) == ("ctrl", "q"):
            if register is not None and register.name == "ctrl":
                return control_register_wire_symbol(register, index or (), (1,))
            result = on_each_wire_symbol(
                step, register, index, width=registers[1].bitsize, child_symbol=x_symbol
            )
            return controlled_wire_title(*result) if register is None else result
        return fallback(step, register, index)

    return supply


def controlled_word_child_decomposition(
    record: ProgramRecord, entry: str
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Qualify complement/repetition control only after whole-definition matching."""
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    ports = tuple(r.name for r in registers)
    if (
        ports not in {("ctrl", "x"), ("ctrl", "q")}
        or registers[0].shape_symbolic
        or registers[1].shape_symbolic
    ):
        return None
    template = controlled_complement_program(registers[1].dtype)
    if ports == ("ctrl", "q"):
        if type(registers[1].bitsize) is not int:
            return None
        callee = template.functions[0].blocks[0].operations[0].callee
        child = next(f for f in template.functions if f.name == callee)
        template = replace(
            template, functions=(child, *(f for f in template.functions if f != child))
        )
    if matches_component_definition(record, entry, template):
        return "Controlled", template, controlled_word_symbols(template)
    return None
