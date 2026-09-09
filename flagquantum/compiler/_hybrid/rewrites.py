"""SSA-preserving rewrite helpers for private hybrid compiler passes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Mapping

from .model import Block, Operation, Region, Value, ValueId


def rewrite_regions(
    operation: Operation,
    rewrite_block: Callable[[Block], Block],
) -> Operation:
    """Apply one block rewrite recursively to all operation regions."""

    regions = tuple(
        Region(tuple(rewrite_block(block) for block in region.blocks))
        for region in operation.regions
    )
    return (
        operation
        if regions == operation.regions
        else replace(operation, regions=regions)
    )


def substitute_value(value: Value, substitutions: Mapping[ValueId, Value]) -> Value:
    """Resolve a transitive SSA substitution and reject accidental cycles."""

    visited: set[ValueId] = set()
    current = value
    while current.id in substitutions:
        if current.id in visited:
            raise ValueError("hybrid SSA substitution contains a cycle")
        visited.add(current.id)
        current = substitutions[current.id]
    return current


def substitute_operation(
    operation: Operation, substitutions: Mapping[ValueId, Value]
) -> Operation:
    """Substitute operands, including values captured by nested regions."""

    operands = tuple(
        substitute_value(value, substitutions) for value in operation.operands
    )
    regions = tuple(
        Region(
            tuple(
                Block(
                    block.arguments,
                    tuple(
                        substitute_operation(nested, substitutions)
                        for nested in block.operations
                    ),
                )
                for block in region.blocks
            )
        )
        for region in operation.regions
    )
    return (
        operation
        if operands == operation.operands and regions == operation.regions
        else replace(operation, operands=operands, regions=regions)
    )


def fresh_value(value: Value, suffix: str) -> Value:
    """Clone a value type under a deterministic fresh SSA scope."""

    return Value(
        ValueId(f"{value.id.scope}.{suffix}", value.id.index),
        value.type,
    )


def clone_inline_block(
    block: Block,
    bindings: Mapping[ValueId, Value],
    *,
    suffix: str,
) -> Block:
    """Clone a block for inlining while freshening all nested definitions."""

    def clone_block(
        nested: Block,
        inherited: Mapping[ValueId, Value],
        argument_bindings: Mapping[ValueId, Value] | None = None,
    ) -> Block:
        substitutions = dict(inherited)
        if argument_bindings is None:
            arguments = tuple(fresh_value(value, suffix) for value in nested.arguments)
            substitutions.update(
                (original.id, fresh)
                for original, fresh in zip(nested.arguments, arguments)
            )
        else:
            arguments = ()
            substitutions.update(argument_bindings)
        operations: list[Operation] = []
        for operation in nested.operations:
            operands = tuple(
                substitute_value(value, substitutions) for value in operation.operands
            )
            results = tuple(fresh_value(value, suffix) for value in operation.results)
            substitutions.update(
                (original.id, fresh)
                for original, fresh in zip(operation.results, results)
            )
            regions = tuple(
                Region(
                    tuple(
                        clone_block(region_block, substitutions)
                        for region_block in region.blocks
                    )
                )
                for region in operation.regions
            )
            operations.append(
                replace(
                    operation,
                    operands=operands,
                    results=results,
                    regions=regions,
                )
            )
        return Block(arguments, tuple(operations))

    return clone_block(block, bindings, bindings)


def region_operation_count(region: Region) -> int:
    """Count operations recursively within one region."""

    return sum(
        1 + sum(region_operation_count(nested) for nested in operation.regions)
        for block in region.blocks
        for operation in block.operations
    )


__all__ = (
    "clone_inline_block",
    "fresh_value",
    "region_operation_count",
    "rewrite_regions",
    "substitute_operation",
    "substitute_value",
)
