"""Deriving the program level's value graph from a public circuit, and back.

`circuit_to_program` is a real lowering. It reads each instruction's opcode
signature, declares one live qubit value per qubit, chains every gate's consumed
value into the successor it defines, and binds every declared parameter to a
`scalar` value defined by a `constant` operation. A measurement additionally
defines one boolean SSA value, reset advances its qubit value, and conditioned
gates read the live measurement values through canonical DNF. The result is a
straight-line single-entry block whose value flow is the circuit's data flow.

`program_to_circuit` is the inverse, and it runs *before* the recorded payload is
restored: the payload is lowered again and its complete canonical function graph
must equal the record's graph. A record whose graph was edited -- a dropped
operation, an operand re-bound to another value, a retyped result, or a changed
condition -- therefore refuses instead of being reported as an exact round trip.
The payload itself is then restored and re-canonicalized, which catches a payload
that decodes but does not re-encode to the recorded form.

The payload is an attestation, not a source. `MULTI_LEVEL_IR_ARCHITECTURE.md`
section 5.5 keeps observables, terminal measurement requests, and request metadata
with the request rather than with the program. Mid-circuit classical-control
metadata is the exception because it is executable semantics and is represented
as typed def-use. The program carries the canonical payload the circuit came from
and states which sections the round trip has to reproduce. A record without one
has nothing to attest against, so it is refused rather than silently
under-restored.
"""

from __future__ import annotations

from typing import Any

from ....errors import ValidationError
from ...classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from .. import CircuitIR, Instruction, IRSerializationError, IRValidationError
from ..diagnostics import LEVEL_PROGRAM, PUBLIC_LEVEL, LevelDiagnostic, _diagnostic
from .model import (
    CONSTANT_OPCODE,
    INVALID_INPUT,
    SUPPORTED_EXACT,
    UNSUPPORTED_WITH_DIAGNOSTICS,
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    LevelConversion,
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

    Static and supported dynamic `CircuitIR` values enter exactly. Dynamic
    metadata is validated before lowering so measurement results and conditions
    can be represented as typed def-use rather than interpreted heuristically.
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
    conditional = any(
        "conditions" in instruction.metadata
        or "condition_clauses" in instruction.metadata
        for instruction in program.instructions
    )
    try:
        validate_classical_control(
            program.instructions,
            n_qubits=program.n_wires,
        )
    except (TypeError, ValidationError) as exc:
        diagnostic = (
            _diagnostic(
                "level.conditional_instruction",
                f"invalid classical-control metadata: {exc}",
            )
            if conditional
            else _diagnostic(
                "level.dynamic_instruction",
                f"invalid classical-control metadata: {exc}",
            )
        )
        return LevelConversion(
            state=INVALID_INPUT,
            level=PUBLIC_LEVEL,
            diagnostics=(diagnostic,),
        )
    for index, instruction in enumerate(program.instructions):
        if instruction.metadata.get("is_dynamic") and instruction.name not in {
            "measure",
            "reset",
        }:
            return LevelConversion(
                state=UNSUPPORTED_WITH_DIAGNOSTICS,
                level=PUBLIC_LEVEL,
                diagnostics=(
                    _diagnostic(
                        "level.dynamic_instruction",
                        f"instruction {index} ({instruction.name!r}) uses the dynamic "
                        "marker outside measurement or reset",
                    ),
                ),
            )
    return LevelConversion(
        state=SUPPORTED_EXACT,
        level=PUBLIC_LEVEL,
        canonical_payload=program.to_dict(),
    )


def _condition_identity(condition: ConditionRef) -> tuple[str, int, bool]:
    return condition.value.scope, condition.value.index, condition.equals


def _lower_conditions(
    instruction: Instruction,
    classical_values: dict[int, ValueRef],
) -> tuple[tuple[ConditionRef, ...], ...]:
    """Map canonical public bit tests to canonical boolean SSA references."""

    clauses = []
    for clause in instruction_condition_clauses(instruction):
        conditions = tuple(
            sorted(
                (
                    ConditionRef(classical_values[bit], bool(expected))
                    for bit, expected in clause
                ),
                key=_condition_identity,
            )
        )
        clauses.append(conditions)
    return tuple(
        sorted(
            clauses,
            key=lambda clause: (
                len(clause),
                tuple(_condition_identity(condition) for condition in clause),
            ),
        )
    )


def _lower_instructions(circuit: CircuitIR) -> BlockRecord:
    """Build the entry block whose operations realize `circuit`'s instructions.

    The block declares one live qubit value per qubit, and each instruction
    consumes the value its qubit currently holds and defines that qubit's successor.
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
    classical_values: dict[int, ValueRef] = {}
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
        results = tuple(builder.define(kind) for kind in signature.result_kinds)
        qubit_results = results[: len(instruction.wires)]
        for wire, result in zip(instruction.wires, qubit_results, strict=True):
            wires[wire] = result
        if instruction.name == "measure":
            classical_values[int(instruction.metadata["classical_bit"])] = results[-1]
        operations.append(
            Operation(
                instruction.name,
                operands=operands,
                results=results,
                parameters=tuple(parameters),
                condition_clauses=_lower_conditions(
                    instruction,
                    classical_values,
                ),
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


def program_to_circuit(record: Any) -> LevelConversion:
    """Restore the public circuit a program-level record states.

    The record is verified, the payload is lowered into its canonical function
    graph, and that graph must agree exactly with the record before the payload is
    restored. The restored circuit must then canonicalize to exactly the payload
    the record carried.
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
    if record.physical_context is not None:
        from .physical import IsingContext

        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.no_canonical_payload",
                    (
                        "native Ising braids cannot lower to CircuitIR"
                        if isinstance(record.physical_context, IsingContext)
                        else "native Rydberg timelines cannot lower to CircuitIR; use flagquantum.lindblad"
                    ),
                ),
            ),
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
    classification = classify_public_circuit(circuit)
    try:
        expected_entry = _lower_instructions(circuit)
    except (KeyError, _UnmodelledOpcodeError):
        expected_entry = None
    expected_functions = (
        ()
        if expected_entry is None
        else (FunctionRecord(name="main", entry="entry", blocks=(expected_entry,)),)
    )
    if (
        classification.state != SUPPORTED_EXACT
        or expected_entry is None
        or record.functions != expected_functions
    ):
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.round_trip_mismatch",
                    "the operation graph does not exactly re-derive the canonical "
                    "instruction payload",
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
