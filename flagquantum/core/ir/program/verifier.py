"""Structural verification of the internal IR levels.

A level verifier answers one question: does this record state a well-typed
program? It returns every refusal it finds rather than raising the first one, so a
caller sees the whole defect set at once, and it never repairs what it reads. The
program level is verified here; the quantum and target levels add their own
functions beside it rather than a second verifier.

The program level's rules come from `MULTI_LEVEL_IR_ARCHITECTURE.md` sections 5.2
and 5.3.1 and from proposal 062 section 3. A value is identified by scope and
index, a linear value has exactly one consumer, a join contributes exactly the
kinds its target declares, and an operation must be well typed against the
operator vocabulary the repository already owns.
"""

from __future__ import annotations

from ..diagnostics import _diagnostic
from .model import (
    CONSTANT_OPCODE,
    TERMINATORS,
    VALUE_KINDS,
    BlockRecord,
    FunctionRecord,
    LevelDiagnostic,
    Operation,
    ProgramRecord,
    ValueRef,
    opcode_signature,
)

__all__ = ("verify_program_record",)


def verify_program_record(record: ProgramRecord) -> tuple[LevelDiagnostic, ...]:
    """Return every structural refusal this program-level record warrants.

    Checks value identity, terminators, control-flow edges, joins, operation
    typing, and linearity. A join is rejected unless every edge into a block
    contributes exactly the kinds that block declares as arguments, which is the
    positional form of "every predecessor contributes exactly one live value for
    the same resource".

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
        diagnostics.extend(_verify_block(block, arguments, known))
    return diagnostics


def _verify_block(
    block: BlockRecord,
    arguments: dict[str, tuple[ValueRef, ...]],
    known: set[str],
) -> list[LevelDiagnostic]:
    """Verify one block, including the values its operations define and consume."""

    diagnostics: list[LevelDiagnostic] = list(_verify_value_identities(block))
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

    #: How many times each value is consumed inside this block. A value is scoped
    #: to the block that defines it, so both kinds of consumer -- a forwarded
    #: edge and an operation operand -- are counted over the same block.
    consumers: dict[ValueRef, int] = {}
    for edge in block.edges:
        if edge.target not in known:
            diagnostics.append(
                _diagnostic(
                    "program.edge_unknown",
                    f"block {block.name!r} branches to unknown block {edge.target!r}",
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

    owned, operation_diagnostics = _verify_operations(block, consumers)
    diagnostics.extend(operation_diagnostics)
    diagnostics.extend(_verify_linearity(block, owned, consumers))
    return diagnostics


def _verify_operations(
    block: BlockRecord, consumers: dict[ValueRef, int]
) -> tuple[tuple[ValueRef, ...], list[LevelDiagnostic]]:
    """Type every operation, and return the linear values the block owns.

    An operation is checked against the signature of its opcode: how many values
    it consumes, what kind each one has, which parameter names it must bind, and
    what it defines. A value it consumes must already be live -- a block argument
    or an earlier operation's result -- and a value it defines must not be, which
    is what turns a stale use or a rebinding into a refusal instead of a silent
    overwrite.
    """

    diagnostics: list[LevelDiagnostic] = []
    live = set(block.arguments)
    results: list[ValueRef] = []
    for position, operation in enumerate(block.operations):
        signature = opcode_signature(operation.opcode)
        if signature is None:
            diagnostics.append(
                _diagnostic(
                    "program.opcode_unknown",
                    f"operation {position} in block {block.name!r} uses opcode "
                    f"{operation.opcode!r}, which this level does not type",
                )
            )
            continue
        diagnostics.extend(_verify_operation_shape(block, position, operation))
        bound = tuple(value for _, value in operation.parameters)
        for value in (*operation.operands, *bound):
            if value not in live:
                diagnostics.append(
                    _diagnostic(
                        "program.value_undefined",
                        f"operation {position} in block {block.name!r} consumes "
                        f"{_value_text(value)}, which is not live there",
                    )
                )
            consumers[value] = consumers.get(value, 0) + 1
        for value in operation.results:
            if value in live:
                diagnostics.append(
                    _diagnostic(
                        "program.value_redefined",
                        f"operation {position} in block {block.name!r} defines "
                        f"{_value_text(value)}, which is already live there",
                    )
                )
        live.update(operation.results)
        results.extend(operation.results)
    return tuple(results), diagnostics


def _verify_operation_shape(
    block: BlockRecord, position: int, operation: Operation
) -> list[LevelDiagnostic]:
    """Type one operation against its opcode signature, ignoring liveness."""

    signature = opcode_signature(operation.opcode)
    assert signature is not None  # caller checked; narrows for the type checker
    diagnostics: list[LevelDiagnostic] = []
    if len(operation.operands) != signature.arity:
        diagnostics.append(
            _diagnostic(
                "program.operand_arity",
                f"opcode {operation.opcode!r} consumes {signature.arity} value(s), but "
                f"operation {position} in block {block.name!r} states "
                f"{len(operation.operands)}",
            )
        )
    else:
        for operand, expected in zip(
            operation.operands, signature.operand_kinds, strict=True
        ):
            if operand.kind != expected:
                diagnostics.append(
                    _diagnostic(
                        "program.operand_kind",
                        f"opcode {operation.opcode!r} consumes a {expected} value, but "
                        f"operation {position} in block {block.name!r} consumes "
                        f"{_value_text(operand)} of kind {operand.kind!r}",
                    )
                )
    declared = tuple(name for name, _ in operation.parameters)
    if declared != signature.parameters:
        diagnostics.append(
            _diagnostic(
                "program.parameter_missing",
                f"opcode {operation.opcode!r} declares parameters "
                f"{list(signature.parameters)}, but operation {position} in block "
                f"{block.name!r} binds {list(declared)}",
            )
        )
    else:
        for (_, value), _ in zip(
            operation.parameters, signature.parameters, strict=True
        ):
            if value.kind != "scalar":
                diagnostics.append(
                    _diagnostic(
                        "program.operand_kind",
                        f"opcode {operation.opcode!r} binds its parameters to scalar "
                        f"values, but operation {position} in block {block.name!r} "
                        f"binds {_value_text(value)} of kind {value.kind!r}",
                    )
                )
    if len(operation.results) != len(signature.result_kinds):
        diagnostics.append(
            _diagnostic(
                "program.result_kind",
                f"opcode {operation.opcode!r} defines "
                f"{len(signature.result_kinds)} result(s), but operation {position} in "
                f"block {block.name!r} defines {len(operation.results)}",
            )
        )
    else:
        for result, expected in zip(
            operation.results, signature.result_kinds, strict=True
        ):
            if result.kind != expected:
                diagnostics.append(
                    _diagnostic(
                        "program.result_kind",
                        f"opcode {operation.opcode!r} defines a {expected} result, but "
                        f"operation {position} in block {block.name!r} defines "
                        f"{_value_text(result)} of kind {result.kind!r}",
                    )
                )
    if operation.opcode == CONSTANT_OPCODE and operation.literal is None:
        diagnostics.append(
            _diagnostic(
                "program.result_kind",
                f"operation {position} in block {block.name!r} defines a compile "
                "constant without stating its value",
            )
        )
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
    block: BlockRecord,
    owned: tuple[ValueRef, ...],
    consumers: dict[ValueRef, int],
) -> list[LevelDiagnostic]:
    """Check that each linear value the block owns has exactly one consumer.

    A block owns its arguments and every value its operations define. A branching
    block consumes a linear value by forwarding it, so it must forward it exactly
    once. A returning block has no successor and its terminator is the consumer,
    which is the `release` position of the architecture's worked example; there,
    only a forward is a defect.
    """
    diagnostics: list[LevelDiagnostic] = []
    for argument in dict.fromkeys((*block.arguments, *owned)):
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


def _value_text(value: ValueRef) -> str:
    return f"{value.scope}:{value.index}"
