"""Shared finite-call graph constructors; no collected test scenarios."""

from dataclasses import replace

from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    circuit_to_program,
)


def explicit_block(circuit: CircuitIR) -> BlockRecord:
    converted = circuit_to_program(circuit)
    assert converted.record is not None, converted.diagnostics
    block = converted.record.functions[0].blocks[0]
    current = list(block.arguments)
    locations = {value: index for index, value in enumerate(current)}
    for operation in block.operations:
        for before, after in zip(operation.operands, operation.results, strict=False):
            if before.kind == "qubit":
                location = locations.pop(before)
                locations[after] = location
                current[location] = after
    return replace(block, return_values=tuple(current))


def wrap_circuit(circuit: CircuitIR, depth: int = 1) -> ProgramRecord:
    leaf = explicit_block(circuit)
    functions = [FunctionRecord("leaf", (leaf,))]
    previous = "leaf"
    for level in range(depth):
        name = "main" if level == depth - 1 else f"layer{level}"
        arguments = tuple(ValueRef("entry", i, "qubit") for i in range(circuit.n_wires))
        outputs = tuple(
            ValueRef("entry", circuit.n_wires + i, "qubit")
            for i in range(circuit.n_wires)
        )
        call = Operation("call", arguments, outputs, callee=previous)
        block = BlockRecord(
            "entry", arguments, operations=(call,), return_values=outputs
        )
        functions.append(FunctionRecord(name, (block,)))
        previous = name
    return ProgramRecord(
        tuple(reversed(functions)), canonical_payload=circuit.to_dict()
    )
