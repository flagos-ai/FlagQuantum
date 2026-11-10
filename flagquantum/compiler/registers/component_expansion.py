# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Selective expansion adapted from Qualtran v0.7.0 CompositeBloq.flatten_once/flatten.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Selectively expand typed component calls in the existing Program graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from itertools import chain

from ...core.ir.program import (
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from .component_graph import _call_entry
from .component_schedule import local_operation_order


class NoComponentExpansionError(ValueError):
    """No selected component supplied a typed call decomposition."""


def _occurrence_labels(
    function: FunctionRecord, instance_ids: tuple[int, ...] | None
) -> tuple[int, ...]:
    count = len(function.blocks[0].operations)
    if instance_ids is None:
        return tuple(range(count))
    if (
        not isinstance(instance_ids, tuple)
        or len(instance_ids) != count
        or any(type(label) is not int or label < 0 for label in instance_ids)
        or len(set(instance_ids)) != count
    ):
        raise ValueError(
            "instance_ids must contain one distinct nonnegative integer per operation"
        )
    return instance_ids


def _has_call_decomposition(
    function: FunctionRecord, functions: Mapping[str, FunctionRecord]
) -> bool:
    if len(function.blocks) != 1:
        return False
    block = function.blocks[0]
    return (
        block.terminator == "return"
        and not block.edges
        and block.return_values is not None
        and all(
            op.opcode == "call" and functions[op.callee or ""].registers is not None
            for op in block.operations
        )
    )


def expand_component_calls_once(
    record: ProgramRecord,
    entry: str = "main",
    *,
    predicate: Callable[[int, Operation], bool] | None = None,
    instance_ids: tuple[int, ...] | None = None,
) -> tuple[ProgramRecord, tuple[int, ...]]:
    """Expand selected entry calls by one typed-call level in source visitation order.

    The returned labels align with returned entry operations. Retained labels
    and output SSA definitions survive. New labels exceed the previous maximum.
    An empty typed call body counts as a decomposition. Primitive bodies remain
    leaves. Invalid modules/labels raise ValueError and predicate errors propagate.
    NoComponentExpansionError signals a pass without any decomposition.
    """
    function = _call_entry(record, entry)
    block = function.blocks[0]
    labels = _occurrence_labels(function, instance_ids)
    functions = {f.name: f for f in record.functions}
    next_label = max(labels, default=-1) + 1
    defined_values = chain(
        block.arguments, chain.from_iterable(op.results for op in block.operations)
    )
    next_value = max((value.index for value in defined_values), default=-1) + 1
    mapped = {value: value for value in block.arguments}
    operations: list[Operation] = []
    emitted_labels: list[int] = []
    did_work = False
    for position in local_operation_order(function, functions, logical_graph=True):
        operation = block.operations[position]
        label = labels[position]
        operands = tuple(mapped[value] for value in operation.operands)
        callee = functions[operation.callee or ""]
        if (
            predicate is None or predicate(label, operation)
        ) and _has_call_decomposition(callee, functions):
            body = callee.blocks[0]
            local = dict(zip(body.arguments, operands, strict=True))
            for child_position in local_operation_order(
                callee, functions, logical_graph=True
            ):
                child = body.operations[child_position]
                results = tuple(
                    ValueRef(block.name, next_value + i, value.kind)
                    for i, value in enumerate(child.results)
                )
                next_value += len(results)
                operations.append(
                    replace(
                        child,
                        operands=tuple(local[value] for value in child.operands),
                        results=results,
                    )
                )
                local.update(zip(child.results, results, strict=True))
                emitted_labels.append(next_label)
                next_label += 1
            assert body.return_values is not None
            mapped.update(
                zip(
                    operation.results,
                    (local[value] for value in body.return_values),
                    strict=True,
                )
            )
            did_work = True
        else:
            operations.append(replace(operation, operands=operands))
            mapped.update((value, value) for value in operation.results)
            emitted_labels.append(label)
    if not did_work:
        raise NoComponentExpansionError("no selected typed call decomposition")
    assert block.return_values is not None
    expanded = replace(
        block,
        operations=tuple(operations),
        return_values=tuple(mapped[value] for value in block.return_values),
    )
    result = replace(
        record,
        functions=tuple(
            replace(f, blocks=(expanded,)) if f.name == entry else f
            for f in record.functions
        ),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid expanded component graph: {diagnostics}")
    return result, tuple(emitted_labels)


def expand_component_calls(
    record: ProgramRecord,
    entry: str = "main",
    *,
    predicate: Callable[[int, Operation], bool] | None = None,
    instance_ids: tuple[int, ...] | None = None,
    max_depth: int = 1_000,
) -> tuple[ProgramRecord, tuple[int, ...]]:
    """Repeat selective one-level expansion until a pass has no decomposition.

    The depth bound includes the terminating no-change probe, matching source
    flatten. Exhaustion raises ValueError, even for zero or negative bounds.
    """
    function = _call_entry(record, entry)
    labels = _occurrence_labels(function, instance_ids)
    for _ in range(max_depth):
        try:
            record, labels = expand_component_calls_once(
                record, entry, predicate=predicate, instance_ids=labels
            )
        except NoComponentExpansionError:
            return record, labels
    raise ValueError("Max recursion depth exceeded in component expansion.")
