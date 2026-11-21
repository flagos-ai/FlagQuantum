# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/bitwise.py and negate.py.

"""Draw word arithmetic through existing executable or declared child Programs."""

from dataclasses import replace
from functools import partial

from flagquantum.compiler.registers.arithmetic.negation_component import (
    negation_program,
)
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
from flagquantum.drawer.bitwise_symbols import bitwise_wire_symbol
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.on_each_symbols import on_each_wire_symbol
from flagquantum.drawer.partition_symbols import (
    partition_wire_symbol,
    split_join_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def word_program(operation: str, ports: Signature) -> ProgramRecord | None:
    """Retain complete immediate calls for XOR, complement and negation.

    Concrete words reuse executable domain owners verbatim. Symbolic complement
    and negation retain external child declarations, as their source immediate
    decompositions do. Those declarations have no executable body. Symbolic XOR
    retains the original decomposition refusal and receives no native graph.
    """
    if operation not in {"Xor", "BitwiseNot", "Negate"}:
        return None
    dtype = ports[0].dtype
    if type(dtype.num_qubits) is int:
        if operation == "Negate":
            return negation_program(dtype)
        return bitwise_word_program(
            dtype, "xor" if operation == "Xor" else "bitwise_not"
        )
    if operation == "Xor":
        return None
    if operation == "BitwiseNot":
        child = FunctionRecord(
            "OnEach_X", (), registers=tuple(Signature.build_from_dtypes(q=dtype))
        )
        builder = ComponentProgramBuilder(ports, (child,))
        result = builder.call_dict(child.name, {"q": builder.inputs["x"]})
        return builder.finalize({"x": result["q"]})
    children = tuple(
        FunctionRecord(name, (), registers=tuple(ports))
        for name in ("BitwiseNot", "AddK_1")
    )
    builder = ComponentProgramBuilder(ports, children)
    value = builder.inputs["x"]
    for child in children:
        value = builder.call_dict(child.name, {"x": value})["x"]
    return builder.finalize({"x": value})


def word_symbols(record: ProgramRecord, operation: str) -> RegisterWireSymbolSupplier:
    """Reuse native bitwise, Pauli, fanout and partition drawing callbacks."""
    functions = {f.name: f for f in record.functions}
    root = record.functions[0]
    assert root.registers is not None
    width = root.registers[0].bitsize
    complement = root.blocks[0].operations[0].callee if operation == "Negate" else None
    x_symbol = partial(pauli_basis_wire_symbol, gate="x")

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if operation == "Negate":
            if step.name == complement:
                return bitwise_wire_symbol(
                    step, register, index, operation="bitwise_not"
                )
            if register is None:
                return "Text", {"text": "AddK", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        function = functions[step.name]
        ports = tuple(r.name for r in function.registers or ())
        if operation in {"OnEach", "XorK"} and ports == ("q",):
            return x_symbol(step, register, index)
        if operation == "BitwiseNot" and ports == ("q",):
            return on_each_wire_symbol(
                replace(step, name=f"X⨂{width}"),
                register,
                index,
                width=width,
                child_symbol=x_symbol,
            )
        if ports == ("ctrl", "target"):
            return controlled_not_wire_symbol(step, register, index)
        return split_join_wire_symbol(step, register, index)

    return supply


def word_score(operation: str, ports: Signature) -> MusicalScoreData | None:
    """Project existing stored calls into the ordinary catalog score format."""
    record = word_program(operation, ports)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=word_symbols(record, operation),
    )
