"""Lower owned register words to existing gate SSA, then expand finite calls.

Core verifies logical element identity, producer types and single consumption.
Compiler only rebinds ordered wires, preserving primitive and feedback order.
"""

from __future__ import annotations

from dataclasses import replace

from ...core.ir.program import (
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from ...core.ir.program.register_verifier import (
    executable_register_width,
    register_port_types,
)
from ...core.ir.program.registers import RegisterSide
from .program_calls import _remap_conditions, inline_program_calls


def lower_register_program(record: ProgramRecord, entry: str = "main") -> ProgramRecord:
    """Expand finite concrete word functions into one gate-level Program body.

    Each declared element contributes its MSB-first wires in row-major port
    order. Zero-width words retain ownership until Core verification, then have
    no physical wires. Split/join emit no gates or physical workspace. Finite
    call expansion reuses the existing Compiler inliner and its SSA maps.

    Raises:
        ValueError: The complete module is invalid or outside the admitted finite
            concrete register profile, or its entry is absent.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid register Program: {diagnostics}")
    if any(
        operation.opcode in {"allocate_register", "free_register"}
        for function in record.functions
        for block in function.blocks
        for operation in block.operations
    ):
        raise ValueError(
            "register workspace lifetimes require physical pool scheduling and "
            "clean/dirty restoration evidence before gate lowering"
        )
    if any(
        operation.opcode
        in {"measure_register", "parity_register", "read_classical_register"}
        for function in record.functions
        for block in function.blocks
        for operation in block.operations
    ):
        raise ValueError(
            "owned classical operation requires physical register binding "
            "before gate lowering"
        )
    functions = {function.name: function for function in record.functions}
    lowered = replace(
        record,
        functions=tuple(
            _lower_function(function, functions) for function in record.functions
        ),
    )
    return inline_program_calls(lowered, entry)


def _lower_function(
    function: FunctionRecord, functions: dict[str, FunctionRecord]
) -> FunctionRecord:
    if function.registers is None:
        return function
    block = function.blocks[0]
    assert block.return_values is not None
    values: dict[ValueRef, tuple[ValueRef, ...]] = {}
    single_values: dict[ValueRef, ValueRef] = {}
    next_index = 0

    def define(value: ValueRef, width: int = 1) -> tuple[ValueRef, ...]:
        nonlocal next_index
        kind = "qubit" if value.kind == "register" else value.kind
        wires = tuple(
            ValueRef(block.name, next_index + index, kind) for index in range(width)
        )
        next_index += width
        values[value] = wires
        if value.kind != "register":
            single_values[value] = wires[0]
        return wires

    arguments = tuple(
        wire
        for value, dtype in zip(
            block.arguments,
            register_port_types(function, RegisterSide.LEFT),
            strict=True,
        )
        for wire in define(value, executable_register_width(dtype))
    )
    operations: list[Operation] = []
    for operation in block.operations:
        operands = tuple(wire for value in operation.operands for wire in values[value])
        if operation.opcode == "split_register":
            single_values.update(zip(operation.results, operands, strict=True))
            values.update(
                (value, (single_values[value],)) for value in operation.results
            )
            continue
        if operation.opcode == "join_register":
            values[operation.results[0]] = operands
            continue
        if operation.opcode == "call":
            callee = functions[operation.callee or ""]
            if callee.registers is not None:
                widths = tuple(
                    executable_register_width(dtype)
                    for dtype in register_port_types(callee, RegisterSide.RIGHT)
                )
            else:
                widths = (1,) * len(operation.results)
        else:
            widths = (1,) * len(operation.results)
        # Parameters and bool conditions are existing scalar SSA, each with one
        # mapped value. Word tuples never become primitive parameter bindings.
        results = tuple(
            wire
            for value, width in zip(operation.results, widths, strict=True)
            for wire in define(value, width)
        )
        operations.append(
            replace(
                operation,
                operands=operands,
                results=results,
                parameters=tuple(
                    (name, single_values[value]) for name, value in operation.parameters
                ),
                condition_clauses=_remap_conditions(
                    operation.condition_clauses, single_values
                ),
            )
        )
    returned = tuple(wire for value in block.return_values for wire in values[value])
    lowered = replace(
        block, arguments=arguments, operations=tuple(operations), return_values=returned
    )
    return replace(function, blocks=(lowered,), registers=None)
