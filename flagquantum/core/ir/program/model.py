"""The program level's model: values, blocks, operations, and their signatures.

`MULTI_LEVEL_IR_ARCHITECTURE.md` section 5.2 states what this level represents --
functions, scopes, blocks and branches, typed arguments, measurement-produced
values -- and section 5.3.1 states the value and linearity rules. This module owns
that model; `verifier.py` enforces it, and `lowering.py` next door derives it from,
and back to, the public IR.

The model is deliberately small. A value is identified by the scope it was defined
in and its index within that scope, never by a name and never by object identity,
so a stale or duplicated use is a comparison rather than a reachability problem. A
block states its arguments, its operations, its terminator, and its successors
explicitly, so the control-flow graph is data rather than an interpreter's state.

Nothing here is re-exported from `flagquantum.core.ir`, recorded in
`docs/public_api_v1.json`, or reachable from `flagquantum`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...operator_schema import get_operator_schema
from ..diagnostics import LevelContractError, LevelDiagnostic

#: `IR-006`'s support states, extended to the internal levels by proposal 062
#: section 3. There is no implicit lossy state and no `best_effort`.
SUPPORTED_EXACT = "supported_exact"
UNSUPPORTED_WITH_DIAGNOSTICS = "unsupported_with_diagnostics"
INVALID_INPUT = "invalid_input"

SUPPORT_STATES = (SUPPORTED_EXACT, UNSUPPORTED_WITH_DIAGNOSTICS, INVALID_INPUT)

#: Closed value-kind vocabulary. `qubit` is the linear kind: a qubit value
#: identifies one resource state at one program point, so it has one consumer.
VALUE_KINDS = ("bool", "index", "scalar", "tensor", "qubit")

LINEAR_VALUE_KINDS = ("qubit",)

TERMINATORS = ("return", "branch")

#: The one program-level opcode that is not a circuit opcode. A parameter is a
#: compile constant that the circuit states as a number and the program states as
#: a `scalar` value, so a gate operation binds a value rather than a literal and
#: the value's def-use is checkable.
CONSTANT_OPCODE = "constant"


@dataclass(frozen=True)
class ValueRef:
    """Identity of one definition, plus the kind that decides its linearity."""

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
class Operation:
    """One program-level operation: what it consumes, defines, and binds.

    `operands` are the values the operation consumes and `results` the values it
    defines, positionally. A circuit opcode defines one successor per consumed
    qubit, so `results` is what makes a stale use visible: consuming a value that
    a later operation has already replaced gives that value two consumers.

    `parameters` binds every parameter the opcode declares to a `scalar` value
    rather than to a number, and `literal` carries the number for the one
    `constant` opcode that defines such a value.

    Attributes:
        opcode: Either `CONSTANT_OPCODE` or a canonical circuit opcode.
        operands: The values this operation consumes, positionally.
        results: The values this operation defines, positionally.
        parameters: Declared parameter name to the `scalar` value binding it.
        literal: The compile constant, present only on `CONSTANT_OPCODE`.
    """

    opcode: str
    operands: tuple[ValueRef, ...] = ()
    results: tuple[ValueRef, ...] = ()
    parameters: tuple[tuple[str, ValueRef], ...] = ()
    literal: float | None = None


@dataclass(frozen=True)
class BlockRecord:
    """A single-entry block with explicit arguments and explicit successors."""

    name: str
    arguments: tuple[ValueRef, ...] = ()
    terminator: str = "return"
    edges: tuple[Edge, ...] = ()
    operations: tuple[Operation, ...] = ()


@dataclass(frozen=True)
class FunctionRecord:
    """One function: a named entry block and the blocks it can reach."""

    name: str
    blocks: tuple[BlockRecord, ...]
    entry: str = "entry"


@dataclass(frozen=True)
class ProgramRecord:
    """Program-level module: one function per record.

    `canonical_payload` is the canonical `CircuitIR.to_dict()` this module was
    lowered from. It is excluded from identity, because the record's operation
    graph is its semantic identity and the payload is the exact form the lowering
    is measured against.
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


@dataclass(frozen=True)
class OpcodeSignature:
    """The kinds one opcode consumes and defines, plus the names it binds.

    A circuit opcode consumes one qubit per wire and defines its successor value
    for the same wire, which is what makes the value flow a chain rather than a
    tree: a channel maps a qubit to its successor just as a unitary does. The one
    opcode that is not a circuit opcode, `CONSTANT_OPCODE`, consumes nothing and
    defines one `scalar`.

    Attributes:
        opcode: The canonical opcode this signature describes.
        parameters: The parameter names the opcode requires, positionally.
        operand_kinds: The kind each consumed value must have, positionally.
        result_kinds: The kind each defined value has, positionally.
    """

    opcode: str
    parameters: tuple[str, ...] = ()
    operand_kinds: tuple[str, ...] = ()
    result_kinds: tuple[str, ...] = ()

    @property
    def arity(self) -> int:
        """The number of operands this opcode consumes."""
        return len(self.operand_kinds)


def opcode_signature(opcode: str) -> OpcodeSignature | None:
    """Return the type signature of one opcode this level models.

    The canonical operator vocabulary is the one the repository already owns, so
    this reads `flagquantum.core.operator_schema` rather than restating it: a
    program-level signature that drifted from the operator manifest would silently
    retype every gate. An opcode outside that vocabulary has no program-level
    signature and is therefore not modelled here; the quantum level owns the
    extended vocabulary.
    """
    if opcode == CONSTANT_OPCODE:
        return OpcodeSignature(opcode, result_kinds=("scalar",))
    schema = get_operator_schema(opcode)
    if schema is None:
        return None
    wires = ("qubit",) * schema.arity
    return OpcodeSignature(
        opcode,
        parameters=tuple(schema.parameters),
        operand_kinds=wires,
        result_kinds=wires,
    )
