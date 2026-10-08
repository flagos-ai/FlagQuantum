"""Deriving the program level's value graph from a public circuit, and back.

`circuit_to_program` is a real lowering. It reads each instruction's opcode
signature, declares one live qubit value per wire, chains every gate's consumed
value into the successor it defines, and binds every declared parameter to a
`scalar` value defined by a `constant` operation. The result is a straight-line
single-entry block whose value flow is the circuit's data flow.

`program_to_circuit` is the inverse, and it runs *before* the recorded payload is
restored: the operation graph is re-derived into an instruction sequence, and that
sequence must agree with the sequence the recorded payload states. A record whose
graph was edited -- a dropped operation, a rewired operand, a retyped result --
therefore refuses instead of being reported as an exact round trip. The payload
itself is then restored and re-canonicalized, which catches a payload that decodes
but does not re-encode to the recorded form.

The payload is an attestation, not a source. `MULTI_LEVEL_IR_ARCHITECTURE.md`
section 5.5 keeps observables, measurements, and metadata with the request rather
than with the program, so the program level does not model them: it carries the
canonical payload the circuit came from and states which sections the round trip
has to reproduce. A record without one has nothing to attest against, so it is
refused rather than silently under-restored.
"""

from __future__ import annotations

from typing import Any

from .. import CircuitIR, IRSerializationError, IRValidationError
from ..diagnostics import LEVEL_PROGRAM, PUBLIC_LEVEL, _diagnostic
from .model import (
    CONSTANT_OPCODE,
    INVALID_INPUT,
    SUPPORTED_EXACT,
    UNSUPPORTED_WITH_DIAGNOSTICS,
    BlockRecord,
    FunctionRecord,
    LevelConversion,
    LevelDiagnostic,
    Operation,
    ProgramRecord,
    ValueRef,
    opcode_signature,
)
from .verifier import verify_program_record

__all__ = (
    "circuit_to_program",
    "classify_public_circuit",
    "program_to_circuit",
)


class _UnmodelledOpcodeError(Exception):
    """One instruction the program level cannot state, carrying its refusal."""

    def __init__(self, diagnostic: LevelDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class _BlockBuilder:
    """Numbers the values one block defines, in definition order."""

    def __init__(self, name: str, arguments: tuple[ValueRef, ...]) -> None:
        self.name = name
        self.arguments = arguments
        self._next = max((value.index for value in arguments), default=-1) + 1

    def define(self, kind: str) -> ValueRef:
        value = ValueRef(self.name, self._next, kind)
        self._next += 1
        return value


def classify_public_circuit(program: Any) -> LevelConversion:
    """Classify one public input against the level boundary.

    Static `CircuitIR` values enter exactly. A dynamic instruction, or one
    carrying declared conditions, is refused with diagnostics because the internal
    levels would need a measurement def-use graph the public schema does not carry.
    Anything else is invalid input.
    """
    if not isinstance(program, CircuitIR):
        return LevelConversion(
            state=INVALID_INPUT,
            level=PUBLIC_LEVEL,
            diagnostics=(
                _diagnostic(
                    "level.not_circuit_ir",
                    f"expected a CircuitIR, found {type(program).__name__}",
                ),
            ),
        )
    for index, instruction in enumerate(program.instructions):
        if instruction.metadata.get("is_dynamic"):
            return LevelConversion(
                state=UNSUPPORTED_WITH_DIAGNOSTICS,
                level=PUBLIC_LEVEL,
                diagnostics=(
                    _diagnostic(
                        "level.dynamic_instruction",
                        f"instruction {index} ({instruction.name!r}) is dynamic and has "
                        "no def-use representation at the internal levels",
                    ),
                ),
            )
        if instruction.metadata.get("conditions"):
            return LevelConversion(
                state=UNSUPPORTED_WITH_DIAGNOSTICS,
                level=PUBLIC_LEVEL,
                diagnostics=(
                    _diagnostic(
                        "level.conditional_instruction",
                        f"instruction {index} ({instruction.name!r}) declares conditions "
                        "that the public schema does not resolve",
                    ),
                ),
            )
    return LevelConversion(
        state=SUPPORTED_EXACT,
        level=PUBLIC_LEVEL,
        canonical_payload=program.to_dict(),
    )


def _lower_instructions(circuit: CircuitIR) -> BlockRecord:
    """Build the entry block whose operations realize `circuit`'s instructions.

    The block declares one live qubit value per wire, and each instruction
    consumes the value its wire currently holds and defines that wire's successor.
    A parameter becomes a `constant` operation defining a `scalar` value that the
    gate operation binds by name, so a parameter is a value with a def-use rather
    than a literal embedded in an operation.

    Raises:
        _UnmodelledOpcodeError: One instruction's opcode has no program-level type
            signature, so the circuit is refused rather than approximated.
    """
    arguments = tuple(
        ValueRef("entry", index, "qubit") for index in range(circuit.n_wires)
    )
    builder = _BlockBuilder("entry", arguments)
    wires = list(arguments)
    operations: list[Operation] = []
    for index, instruction in enumerate(circuit.instructions):
        signature = opcode_signature(instruction.name)
        if signature is None:
            raise _UnmodelledOpcodeError(
                _diagnostic(
                    "program.opcode_unknown",
                    f"instruction {index} ({instruction.name!r}) is outside the program "
                    "level's operator vocabulary, so its operands and results are not "
                    "typed here",
                )
            )
        parameters: list[tuple[str, ValueRef]] = []
        for name in signature.parameters:
            scalar = builder.define("scalar")
            operations.append(
                Operation(
                    CONSTANT_OPCODE,
                    results=(scalar,),
                    literal=float(instruction.params[name]),
                )
            )
            parameters.append((name, scalar))
        operands = tuple(wires[wire] for wire in instruction.wires)
        results = tuple(builder.define("qubit") for _ in instruction.wires)
        for wire, result in zip(instruction.wires, results, strict=True):
            wires[wire] = result
        operations.append(
            Operation(
                instruction.name,
                operands=operands,
                results=results,
                parameters=tuple(parameters),
            )
        )
    return BlockRecord("entry", arguments, operations=tuple(operations))


def circuit_to_program(program: Any) -> LevelConversion:
    """Lower one public circuit to the program level, exactly or not at all.

    The record states the circuit's control-flow graph, its typed value flow, and
    the canonical payload the lowering is measured against. Nothing about the
    payload is copied into the graph: the graph is derived from the instructions,
    and `program_to_circuit` re-derives the instructions from the graph.
    """
    classification = classify_public_circuit(program)
    if classification.state != SUPPORTED_EXACT:
        return LevelConversion(
            state=classification.state,
            level=LEVEL_PROGRAM,
            diagnostics=classification.diagnostics,
        )
    # `classify_public_circuit` proved this is a `CircuitIR` before it returned an
    # exact classification, so the lowering may read it as one.
    try:
        entry = _lower_instructions(program)
    except _UnmodelledOpcodeError as refusal:
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(refusal.diagnostic,),
        )
    record = ProgramRecord(
        functions=(FunctionRecord(name="main", entry=entry.name, blocks=(entry,)),),
        canonical_payload=classification.canonical_payload,
    )
    return LevelConversion(
        state=SUPPORTED_EXACT,
        level=LEVEL_PROGRAM,
        record=record,
        canonical_payload=classification.canonical_payload,
    )


def _reproduce_instructions(entry: BlockRecord) -> list[tuple[str, tuple[int, ...]]]:
    """Re-derive the instruction sequence one block's value graph states.

    Returns one `(opcode, wire indices)` pair per non-constant operation, in
    order. A wire index is recovered from the position of the value the operand
    holds, which is the inverse of the numbering `_lower_instructions` performs.
    An operation that consumes a value the block never defined, or that pairs a
    different number of operands and results, ends the derivation: the caller
    compares the shorter sequence with the payload and refuses.
    """
    wire_of: dict[ValueRef, int] = {
        value: index
        for index, value in enumerate(entry.arguments)
        if value.kind == "qubit"
    }
    reproduced: list[tuple[str, tuple[int, ...]]] = []
    for operation in entry.operations:
        if operation.opcode == CONSTANT_OPCODE:
            continue
        if len(operation.operands) != len(operation.results):
            return reproduced
        try:
            indices = tuple(wire_of[operand] for operand in operation.operands)
        except KeyError:
            return reproduced
        reproduced.append((operation.opcode, indices))
        for operand, result in zip(operation.operands, operation.results, strict=True):
            wire_of[result] = wire_of.pop(operand)
    return reproduced


def program_to_circuit(record: Any) -> LevelConversion:
    """Restore the public circuit a program-level record states.

    The record is verified, its value graph is re-derived into an instruction
    sequence, and that sequence must agree with the payload's instructions before
    the payload is restored. The restored circuit must then canonicalize to
    exactly the payload the record carried.
    """
    if not isinstance(record, ProgramRecord):
        return LevelConversion(
            state=INVALID_INPUT,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.not_program_record",
                    f"expected a ProgramRecord, found {type(record).__name__}",
                ),
            ),
        )
    diagnostics = verify_program_record(record)
    if diagnostics:
        return LevelConversion(
            state=INVALID_INPUT, level=LEVEL_PROGRAM, diagnostics=diagnostics
        )
    if record.canonical_payload is None:
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.no_canonical_payload",
                    "this record records no canonical payload, so the public circuit "
                    "cannot be restored from the operation graph alone",
                ),
            ),
        )
    try:
        circuit = CircuitIR.from_dict(record.canonical_payload)
    except (
        IRValidationError,
        IRSerializationError,
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:
        # The recorded payload is untrusted at this point: it may have been
        # edited since it was recorded. The decoder reaches into nested fields,
        # so a malformed one surfaces as an attribute or key error rather than a
        # validation error, and every one of those is a refusal, not a crash.
        return LevelConversion(
            state=INVALID_INPUT,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.payload_invalid",
                    f"the recorded payload is not a readable CircuitIR: {exc}",
                ),
            ),
        )
    stated = [
        (instruction.name, tuple(instruction.wires))
        for instruction in circuit.instructions
    ]
    function = record.functions[0]
    entry = next(block for block in function.blocks if block.name == function.entry)
    reproduced = _reproduce_instructions(entry)
    if reproduced != stated:
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.round_trip_mismatch",
                    "the operation graph states the instruction sequence "
                    f"{reproduced}, but the recorded payload states {stated}",
                ),
            ),
        )
    restored = circuit.to_dict()
    if restored != record.canonical_payload:
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.round_trip_mismatch",
                    "the recorded payload is not the canonical form: restoring it "
                    "produces a different payload",
                ),
            ),
        )
    return LevelConversion(
        state=SUPPORTED_EXACT, level=PUBLIC_LEVEL, canonical_payload=restored
    )
