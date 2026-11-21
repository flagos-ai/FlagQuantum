# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/subtraction.py.

"""Draw complete subtraction catalog calls through existing native owners."""

from dataclasses import replace
from functools import partial
from typing import cast

from examples.algorithms.subtraction_catalog import symbolic_subtraction_steps
from examples.algorithms.word_arithmetic_decomposition import word_program
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import OnEachCall
from flagquantum.compiler.registers.arithmetic.subtraction_component import (
    subtract_from_program,
    subtraction_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.register_cast import cast_signature
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.addition_symbols import addition_wire_symbol
from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
from flagquantum.drawer.bitwise_symbols import bitwise_wire_symbol
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.on_each_symbols import on_each_wire_symbol
from flagquantum.drawer.partition_symbols import (
    partition_wire_symbol,
    split_join_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    PortSide,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def _symbolic_subtract(ports: Signature) -> ProgramRecord:
    dtype = ports[0].dtype
    width = dtype.num_qubits
    unsigned = unsigned_register_type(width)
    builder = ComponentProgramBuilder(ports)
    values = dict(builder.inputs)
    defined: set[str] = set()
    for child, addresses in symbolic_subtraction_steps(width):
        if isinstance(child, tuple):
            source, target = (
                (dtype, unsigned) if child[1] == "signed" else (unsigned, dtype)
            )
            name = "cast_to_" + str(child[2])
            signature = cast_signature(source, target)
        elif isinstance(child, OnEachCall):
            name = "OnEach_X"
            signature = Signature.build_from_dtypes(q=opaque_register_type(width))
        else:
            assert isinstance(child, ArithmeticCall) and child.operation == "add"
            name = "Add"
            signature = Signature.build_from_dtypes(a=unsigned, b=unsigned)
        if name not in defined:
            builder.define(FunctionRecord(name, (), registers=tuple(signature)))
            defined.add(name)
        result = builder.call_dict(
            name,
            {
                register.name: values[address]
                for register, address in zip(signature.lefts(), addresses, strict=True)
            },
        )
        for register, address in zip(signature.rights(), addresses, strict=True):
            values[address] = result[register.name]
    return builder.finalize(values)


def subtraction_catalog_program(
    operation: str, ports: Signature
) -> ProgramRecord | None:
    """Retain executable concrete owners and declared symbolic children.

    Signed symbolic Subtract reuses the existing source-ordered catalog steps.
    SubtractFrom retains complement/Add/complement and reuses the symbolic
    complement graph. Source refusal leaves retain typed declarations. CompositeBloq keeps the original UI
    decomposition failure boundary and its atomic interface.
    """
    if operation not in {"Subtract", "SubtractFrom"}:
        return None
    dtype = ports[0].dtype
    if type(dtype.num_qubits) is int:
        return (
            subtraction_program(dtype, ports[1].dtype)
            if operation == "Subtract"
            else subtract_from_program(dtype)
        )
    if operation == "Subtract":
        return _symbolic_subtract(ports)
    complement = word_program("BitwiseNot", Signature.build_from_dtypes(x=dtype))
    assert complement is not None
    addition = FunctionRecord("Add", (), registers=tuple(ports))
    builder = ComponentProgramBuilder(ports, (addition,))
    a, b = builder.inputs["a"], builder.inputs["b"]
    b = builder.call_program(complement, {"x": b})["x"]
    result = builder.call_dict(addition.name, {"a": a, "b": b})
    b = builder.call_program(complement, {"x": result["b"]})["x"]
    return builder.finalize({"a": result["a"], "b": b})


def subtraction_symbols(record: ProgramRecord) -> RegisterWireSymbolSupplier:
    """Reuse typed cast, Add, complement, repeated-X and partition symbols."""
    functions = {f.name: f for f in record.functions}
    x_symbol = partial(pauli_basis_wire_symbol, gate="x")

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        registers = function.registers or ()
        ports = tuple(r.name for r in registers)
        opcodes = {op.opcode for b in function.blocks for op in b.operations}
        if (
            ports == ("reg", "reg")
            and registers[0].shape_symbolic == registers[1].shape_symbolic
        ):
            if register is None:
                return "Text", {"text": "Cast", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        if ports == ("a", "b"):
            return addition_wire_symbol(
                replace(step, gate="add"),
                None if register is None else register.name,
                None if register is None else cast(PortSide, register.side.name),
            )
        if ports == ("x",):
            return bitwise_wire_symbol(step, register, index, operation="bitwise_not")
        if ports == ("q",):
            width = registers[0].bitsize
            return on_each_wire_symbol(
                replace(step, name=f"X⨂{width}"),
                register,
                index,
                width=width,
                child_symbol=x_symbol,
            )
        if ports == ("control", "targets"):
            if register is None:
                return "Text", {"text": "MultiTargetCNOT", "fontsize": 10}
            return (
                ("Circle", {"filled": True})
                if register.name == "control"
                else ("ModPlus", {})
            )
        if "allocate_register" in opcodes or "free_register" in opcodes:
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            symbol, _ = partition_wire_symbol(step, register, index)
            return symbol, {
                "text": "alloc" if register.side == RegisterSide.RIGHT else "free"
            }
        return split_join_wire_symbol(step, register, index)

    return supply


def subtraction_score(operation: str, ports: Signature) -> MusicalScoreData | None:
    """Project the complete immediate source layer into the existing Drawer."""
    record = subtraction_catalog_program(operation, ports)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=subtraction_symbols(record),
    )
