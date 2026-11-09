"""Expand finite calls through the existing Program SSA and gate vocabulary.

Core owns validation and returned-value semantics. This module only transforms
that shared graph, retaining primitive order, parameters and boolean feedback.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from ...core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from ...core.ir.program.lowering import _condition_identity


@dataclass
class _CallFrame:
    operations: Iterator[Operation]
    values: dict[ValueRef, ValueRef]
    returned: tuple[ValueRef, ...]
    targets: tuple[ValueRef, ...]


def inline_program_calls(record: ProgramRecord, entry: str = "main") -> ProgramRecord:
    """Expand a verified finite gate-call module into its entry's primitive body.

    Parameters, conditions and explicit returns are remapped in each call's own
    SSA environment. All definitions are verified before unreachable ones are
    omitted. The returned record retains its canonical payload for independent
    Core attestation. It introduces no physical workspace or numerical actions.

    Raises:
        ValueError: The input is invalid, the entry is absent, or its structure
            is outside the finite single-block gate-call profile.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid Program calls: {diagnostics}")
    if any(function.registers is not None for function in record.functions):
        raise ValueError(
            "packed functions require lower_register_program before inlining"
        )
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown Program entry {entry!r}")
    function = functions[entry]
    if record.physical_context is not None or len(function.blocks) != 1:
        raise ValueError("call expansion requires one gate-level entry block")
    block = function.blocks[0]
    if block.terminator != "return" or block.edges or block.return_values is None:
        raise ValueError("call expansion requires explicit returned values")
    operations, returned = _expand(block, functions)
    expanded = replace(block, operations=operations, return_values=returned)
    result = replace(record, functions=(replace(function, blocks=(expanded,)),))
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid expanded Program calls: {diagnostics}")
    return result


def _expand(
    block: BlockRecord,
    functions: dict[str, FunctionRecord],
) -> tuple[tuple[Operation, ...], tuple[ValueRef, ...]]:
    # This explicit stack also handles finite modules deeper than Python's
    # recursion limit. SSA maps are local to invocations, even for reused leaves.
    assert block.return_values is not None
    frames = [
        _CallFrame(
            iter(block.operations),
            {value: value for value in block.arguments},
            block.return_values,
            (),
        )
    ]
    next_index = max((value.index for value in block.arguments), default=-1) + 1
    emitted: list[Operation] = []
    returned: tuple[ValueRef, ...] = ()
    while frames:
        frame = frames[-1]
        operation = next(frame.operations, None)
        if operation is None:
            returned = tuple(frame.values[value] for value in frame.returned)
            frames.pop()
            if frames:
                frames[-1].values.update(zip(frame.targets, returned, strict=True))
            continue
        operands = tuple(frame.values[value] for value in operation.operands)
        if operation.opcode == "call":
            callee = functions[operation.callee or ""]
            body = callee.blocks[0]
            assert body.return_values is not None
            frames.append(
                _CallFrame(
                    iter(body.operations),
                    dict(zip(body.arguments, operands, strict=True)),
                    body.return_values,
                    operation.results,
                )
            )
            continue
        clauses = _remap_conditions(operation.condition_clauses, frame.values)
        if operation.condition_clauses and not clauses:
            # Aliased classical arguments can make every conjunction false.
            # Such a conditional gate forwards its quantum values unchanged.
            frame.values.update(zip(operation.results, operands, strict=True))
            continue
        results = tuple(
            ValueRef(block.name, next_index + index, value.kind)
            for index, value in enumerate(operation.results)
        )
        next_index += len(results)
        emitted.append(
            replace(
                operation,
                operands=operands,
                results=results,
                parameters=tuple(
                    (name, frame.values[value]) for name, value in operation.parameters
                ),
                condition_clauses=clauses,
            )
        )
        frame.values.update(zip(operation.results, results, strict=True))
    return tuple(emitted), returned


def _remap_conditions(
    clauses: tuple[tuple[ConditionRef, ...], ...],
    values: dict[ValueRef, ValueRef],
) -> tuple[tuple[ConditionRef, ...], ...]:
    normalized: set[tuple[ConditionRef, ...]] = set()
    for clause in clauses:
        expected: dict[ValueRef, bool] = {}
        for condition in clause:
            value = values[condition.value]
            if value in expected and expected[value] != condition.equals:
                break
            expected[value] = condition.equals
        else:
            normalized.add(
                tuple(
                    sorted(
                        (
                            ConditionRef(value, equals)
                            for value, equals in expected.items()
                        ),
                        key=_condition_identity,
                    )
                )
            )
    return tuple(
        sorted(
            normalized,
            key=lambda clause: (len(clause), tuple(map(_condition_identity, clause))),
        )
    )
