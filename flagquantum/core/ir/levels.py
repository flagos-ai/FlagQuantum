"""Internal IR level boundary: its vocabulary, and the contract fake.

The public IR is `CircuitIR` schema 1.0. Phase 4 adds program, quantum, and
target levels behind it. Nothing in this module is public: it is not re-exported
from `flagquantum.core.ir`, is absent from `docs/public_api_v1.json`, and adds
no root export.

`contracts/multi-level-ir-internal-v1-candidate.json` owns the boundary's seven
claims -- level entry and exit, value identity, linearity, joins, failure
categories, round trip, and rejection -- and
`tools/check_multi_level_ir_contract.py` is its reader. This module supplies the
two things that contract names:

* the vocabulary the claims are spelled in -- `INTERNAL_LEVELS`,
  `SUPPORT_STATES`, `VALUE_KINDS`, `TERMINATORS`, and `DIAGNOSTIC_LEVELS`;
* the contract fake, `ProgramRecord` with `lower_to_level`, which is an
  in-memory carrier that performs no lowering. It exists so the boundary can be
  conformance-tested before the real levels are implemented, exactly as
  `LoopbackTransport` exercises the realtime protocol without performing I/O.
  The real program, quantum, and target modules are required to pass the same
  suite, `tests/unit/test_multi_level_ir_contract.py`.

The fake is deliberately not a level compiler. It records the canonical
`CircuitIR` payload instead of reconstructing it, and it refuses every
conversion it cannot state exactly -- dynamic structure, conditional regions,
and any record without a recorded payload -- rather than approximating one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from . import CircuitIR, IRSerializationError, IRValidationError

PUBLIC_LEVEL = "circuit"
LEVEL_PROGRAM = "program"
LEVEL_QUANTUM = "quantum"
LEVEL_TARGET = "target"

#: The levels Phase 4 adds behind the public IR. Every one of them is internal.
INTERNAL_LEVELS = (LEVEL_PROGRAM, LEVEL_QUANTUM, LEVEL_TARGET)

SUPPORTED_EXACT = "supported_exact"
UNSUPPORTED_WITH_DIAGNOSTICS = "unsupported_with_diagnostics"
INVALID_INPUT = "invalid_input"

#: `IR-006`'s support states, extended to the internal levels. There is no
#: implicit lossy state and no `best_effort`.
SUPPORT_STATES = (SUPPORTED_EXACT, UNSUPPORTED_WITH_DIAGNOSTICS, INVALID_INPUT)

#: Closed value-kind vocabulary. `qubit` is the linear kind: a qubit value
#: identifies one resource state at one program point, so it has one consumer.
VALUE_KINDS = ("bool", "index", "scalar", "tensor", "qubit")

LINEAR_VALUE_KINDS = ("qubit",)

TERMINATORS = ("return", "branch")

#: Where each declared diagnostic is raised. A code emitted by this module but
#: absent here is a defect, and the gate scans the module for emitted literals
#: so that registering a code is not the same act as being able to raise it.
DIAGNOSTIC_LEVELS: Mapping[str, str] = {
    "level.not_circuit_ir": PUBLIC_LEVEL,
    "level.dynamic_instruction": PUBLIC_LEVEL,
    "level.conditional_instruction": PUBLIC_LEVEL,
    "level.round_trip_mismatch": LEVEL_PROGRAM,
    "level.not_program_record": LEVEL_PROGRAM,
    "level.no_canonical_payload": LEVEL_PROGRAM,
    "level.unsupported_exit": LEVEL_PROGRAM,
    "level.payload_invalid": LEVEL_PROGRAM,
    "program.function_arity": LEVEL_PROGRAM,
    "program.block_missing": LEVEL_PROGRAM,
    "program.block_duplicate": LEVEL_PROGRAM,
    "program.entry_missing": LEVEL_PROGRAM,
    "program.terminator_unknown": LEVEL_PROGRAM,
    "program.edge_missing": LEVEL_PROGRAM,
    "program.edge_unexpected": LEVEL_PROGRAM,
    "program.edge_unknown": LEVEL_PROGRAM,
    "program.edge_arity": LEVEL_PROGRAM,
    "program.edge_kind": LEVEL_PROGRAM,
    "value.identity": LEVEL_PROGRAM,
    "value.linear_duplicate": LEVEL_PROGRAM,
    "value.linear_unconsumed": LEVEL_PROGRAM,
}

#: The exits this fake realizes. `lower_to_level` refuses every other target by
#: name rather than returning an approximation of it.
REALIZED_EXITS = (LEVEL_PROGRAM,)


class LevelContractError(ValueError):
    """Raised when a caller asks the boundary for something it does not define."""


@dataclass(frozen=True)
class LevelDiagnostic:
    """One typed refusal, classified by the level that raised it.

    Attributes:
        code: Dotted name from `DIAGNOSTIC_LEVELS`.
        message: Human-readable statement of what was refused.
        level: The level at which the refusal happened.
    """

    code: str
    message: str
    level: str


@dataclass(frozen=True)
class ValueRef:
    """Identity of one definition, plus the kind that decides its linearity.

    A value is identified by the scope it was defined in and its index within
    that scope, never by a name and never by an object identity.
    """

    scope: str
    index: int
    kind: str = "scalar"

    @property
    def linear(self) -> bool:
        """Whether this value identifies a resource that cannot be copied."""
        return self.kind in LINEAR_VALUE_KINDS


@dataclass(frozen=True)
class Edge:
    """One control-flow edge and the values it contributes to its target."""

    target: str
    values: tuple[ValueRef, ...] = ()


@dataclass(frozen=True)
class BlockRecord:
    """A single-entry block with explicit arguments and explicit successors."""

    name: str
    arguments: tuple[ValueRef, ...] = ()
    terminator: str = "return"
    edges: tuple[Edge, ...] = ()


@dataclass(frozen=True)
class FunctionRecord:
    """One function: a named entry block and the blocks it can reach."""

    name: str
    blocks: tuple[BlockRecord, ...]
    entry: str = "entry"


@dataclass(frozen=True)
class ProgramRecord:
    """Program-level module: one function per record.

    `canonical_payload` is the recorded canonical `CircuitIR.to_dict()` this
    module was carried from. The fake records it because it performs no
    lowering; the real program level is required to produce the same payload by
    construction. It is excluded from identity, because the record's structure
    is its semantic identity and the payload is what the round trip restores.
    """

    functions: tuple[FunctionRecord, ...]
    canonical_payload: Mapping[str, Any] | None = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True)
class LevelConversion:
    """The outcome of crossing, or refusing to cross, a level boundary.

    Attributes:
        state: Exactly one of `SUPPORT_STATES`.
        level: The level that produced this outcome.
        record: The program-level module, present only when one was produced.
        canonical_payload: The canonical `CircuitIR.to_dict()` this result
            carries. Present when `state` is `supported_exact`, and excluded
            from equality because it is a record rather than an identity.
        diagnostics: Typed refusals, empty when `state` is `supported_exact`.
    """

    state: str
    level: str
    record: ProgramRecord | None = None
    canonical_payload: Mapping[str, Any] | None = field(
        default=None, compare=False, hash=False
    )
    diagnostics: tuple[LevelDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in SUPPORT_STATES:
            raise LevelContractError(f"unknown support state {self.state!r}")
        if self.state == SUPPORTED_EXACT and self.diagnostics:
            raise LevelContractError(
                "a supported_exact result cannot carry diagnostics"
            )
        if self.state != SUPPORTED_EXACT and not self.diagnostics:
            raise LevelContractError(
                f"a {self.state} result must state at least one diagnostic"
            )


def _diagnostic(code: str, message: str) -> LevelDiagnostic:
    """Build one diagnostic, taking its level from the declared registry."""
    try:
        level = DIAGNOSTIC_LEVELS[code]
    except KeyError:  # pragma: no cover - a defect, not a caller error
        raise LevelContractError(f"undeclared diagnostic code {code!r}") from None
    return LevelDiagnostic(code=code, message=message, level=level)


def _value_text(value: ValueRef) -> str:
    return f"{value.scope}:{value.index}"


def verify_program_record(record: ProgramRecord) -> tuple[LevelDiagnostic, ...]:
    """Return every structural refusal this program-level record warrants.

    Checks value identity, terminator vocabulary, control-flow edges, joins, and
    linearity. A join is rejected unless every edge into a block contributes
    exactly the kinds that block declares as arguments, which is the positional
    form of "every predecessor contributes exactly one live value for the same
    resource". The fake models block arguments and edges; the real level extends
    the same rules to operation results inside a block.

    This module is internal, so it states no doctest example: the repository's
    example gate runs the docstrings of public entries only, and an example here
    would look tested while nothing ran it.
    """
    diagnostics: list[LevelDiagnostic] = []
    if not record.functions:
        diagnostics.append(
            _diagnostic("program.function_arity", "a program record needs one function")
        )
    for function in record.functions:
        diagnostics.extend(_verify_function(function))
    return tuple(diagnostics)


def _verify_function(function: FunctionRecord) -> list[LevelDiagnostic]:
    diagnostics: list[LevelDiagnostic] = []
    if not function.blocks:
        diagnostics.append(
            _diagnostic(
                "program.block_missing", f"function {function.name!r} has no block"
            )
        )
        return diagnostics

    names = [block.name for block in function.blocks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        diagnostics.append(
            _diagnostic("program.block_duplicate", f"duplicate block(s) {duplicates}")
        )
    known = set(names)
    if function.entry not in known:
        diagnostics.append(
            _diagnostic(
                "program.entry_missing",
                f"function {function.name!r} entry {function.entry!r} is not a block",
            )
        )

    arguments = {block.name: block.arguments for block in function.blocks}
    for block in function.blocks:
        diagnostics.extend(_verify_value_identities(block))
        if block.terminator not in TERMINATORS:
            diagnostics.append(
                _diagnostic(
                    "program.terminator_unknown",
                    f"block {block.name!r} ends with {block.terminator!r}",
                )
            )
        if block.terminator == "branch" and not block.edges:
            diagnostics.append(
                _diagnostic(
                    "program.edge_missing",
                    f"branching block {block.name!r} declares no successor",
                )
            )
        if block.terminator == "return" and block.edges:
            diagnostics.append(
                _diagnostic(
                    "program.edge_unexpected",
                    f"returning block {block.name!r} declares a successor",
                )
            )
        consumers: dict[ValueRef, int] = {}
        for edge in block.edges:
            if edge.target not in known:
                diagnostics.append(
                    _diagnostic(
                        "program.edge_unknown",
                        f"block {block.name!r} branches to unknown block "
                        f"{edge.target!r}",
                    )
                )
                continue
            target = arguments[edge.target]
            if len(edge.values) != len(target):
                diagnostics.append(
                    _diagnostic(
                        "program.edge_arity",
                        f"block {block.name!r} contributes {len(edge.values)} value(s) "
                        f"to {edge.target!r}, which declares {len(target)} argument(s)",
                    )
                )
            elif tuple(value.kind for value in edge.values) != tuple(
                value.kind for value in target
            ):
                diagnostics.append(
                    _diagnostic(
                        "program.edge_kind",
                        f"block {block.name!r} contributes kinds "
                        f"{[value.kind for value in edge.values]} to {edge.target!r}, "
                        f"which declares {[value.kind for value in target]}",
                    )
                )
            for value in edge.values:
                consumers[value] = consumers.get(value, 0) + 1
        diagnostics.extend(_verify_linearity(block, consumers))
    return diagnostics


def _verify_value_identities(block: BlockRecord) -> list[LevelDiagnostic]:
    diagnostics: list[LevelDiagnostic] = []
    for value in block.arguments:
        if not value.scope or value.index < 0:
            diagnostics.append(
                _diagnostic(
                    "value.identity",
                    f"block {block.name!r} declares an unidentifiable value "
                    f"{value.scope!r}:{value.index}",
                )
            )
        elif value.kind not in VALUE_KINDS:
            diagnostics.append(
                _diagnostic(
                    "value.identity",
                    f"block {block.name!r} declares unknown kind {value.kind!r}",
                )
            )
    return diagnostics


def _verify_linearity(
    block: BlockRecord, consumers: Mapping[ValueRef, int]
) -> list[LevelDiagnostic]:
    """Check that each linear argument of `block` has exactly one consumer.

    A branching block consumes a linear argument by forwarding it, so it must
    forward it exactly once. A returning block has no successor and its
    terminator is the consumer, which is the `release` position of the
    architecture's worked example; there, only a forward is a defect.
    """
    diagnostics: list[LevelDiagnostic] = []
    for argument in block.arguments:
        if not argument.linear:
            continue
        count = consumers.get(argument, 0)
        if count > 1:
            diagnostics.append(
                _diagnostic(
                    "value.linear_duplicate",
                    f"linear value {_value_text(argument)} in block {block.name!r} has "
                    f"{count} consumers",
                )
            )
        elif count == 0 and block.terminator == "branch":
            diagnostics.append(
                _diagnostic(
                    "value.linear_unconsumed",
                    f"linear value {_value_text(argument)} in block {block.name!r} is "
                    "never consumed",
                )
            )
    return diagnostics


def classify_public_circuit(program: Any) -> LevelConversion:
    """Classify one public input against the level boundary.

    Static `CircuitIR` values enter exactly. A dynamic instruction, or one
    carrying declared conditions, is refused with diagnostics because the
    internal levels would need a measurement def-use graph the public schema
    does not carry. Anything else is invalid input.
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


def circuit_to_program(program: Any) -> LevelConversion:
    """Carry one public circuit to the program level, exactly or not at all.

    The fake records the canonical payload in a single-block, single-function
    record. It lowers nothing, so a refusal here is a refusal of the input, not
    of the record.
    """
    classification = classify_public_circuit(program)
    if classification.state != SUPPORTED_EXACT:
        return replace(classification, level=LEVEL_PROGRAM)
    record = ProgramRecord(
        functions=(
            FunctionRecord(
                name="main",
                entry="entry",
                blocks=(BlockRecord(name="entry", terminator="return"),),
            ),
        ),
        canonical_payload=classification.canonical_payload,
    )
    return LevelConversion(
        state=SUPPORTED_EXACT,
        level=LEVEL_PROGRAM,
        record=record,
        canonical_payload=classification.canonical_payload,
    )


def program_to_circuit(record: Any) -> LevelConversion:
    """Restore the public circuit a program-level record was carried from.

    The restored circuit must canonicalize to exactly the payload the record
    carried. A payload that decodes but re-encodes differently -- a hand-built
    payload that omits a section the canonical form states -- is a refusal, not a
    silent repair, because the difference would otherwise be reported as an exact
    round trip.
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
                    "cannot be restored without lowering",
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


def lower_to_level(program: Any, target_level: str) -> LevelConversion:
    """Cross the boundary towards `target_level`, refusing unbuilt exits.

    Only `program` is realized here. The quantum and target levels are declared
    by the contract and are refused by name until the implementation that owns
    them lands, so a caller never receives an approximation of a level that does
    not exist.
    """
    if target_level in (LEVEL_QUANTUM, LEVEL_TARGET):
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.unsupported_exit",
                    f"the {target_level!r} level is declared but not implemented",
                ),
            ),
        )
    if target_level in REALIZED_EXITS:
        return circuit_to_program(program)
    if target_level == PUBLIC_LEVEL:
        return classify_public_circuit(program)
    raise LevelContractError(f"unknown target level {target_level!r}")


def round_trip(program: Any) -> LevelConversion:
    """Carry a public circuit across the program level and back again.

    Returns the restore result, with kind `circuit` and the canonical payload the
    input started from, when both directions are exact. Exactness is enforced
    where the payload is restored, so a payload the boundary cannot re-canonicalize
    is refused rather than reported as an exact round trip.
    """
    forward = circuit_to_program(program)
    if forward.state != SUPPORTED_EXACT or forward.record is None:
        return forward
    return program_to_circuit(forward.record)
