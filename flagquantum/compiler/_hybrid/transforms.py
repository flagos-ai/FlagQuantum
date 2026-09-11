"""Concrete transformations for the private hybrid program IR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from .analysis import (
    FOLDABLE_OPERATIONS,
    HybridProgramAnalysis,
    constant_loop_iterations,
)
from .model import Block, HybridProgram, Operation, Region, Value, ValueId
from .rewrites import (
    clone_inline_block,
    fresh_value,
    region_operation_count,
    rewrite_regions,
    substitute_operation,
    substitute_value,
)


@dataclass(frozen=True)
class PassOutcome:
    """One transformed program plus pass-local, non-semantic audit evidence."""

    program: HybridProgram
    statistics: Mapping[str, int] = field(default_factory=dict)
    remarks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        statistics = {str(key): int(value) for key, value in self.statistics.items()}
        if any(value < 0 for value in statistics.values()):
            raise ValueError("pass statistics must be non-negative")
        remarks = tuple(str(item).strip() for item in self.remarks)
        if any(not item for item in remarks):
            raise ValueError("pass remarks must be nonempty")
        object.__setattr__(self, "statistics", MappingProxyType(statistics))
        object.__setattr__(self, "remarks", remarks)


@dataclass(frozen=True)
class ConstantFoldPass:
    """Replace constant-only arithmetic with typed constants."""

    name: str = "constant_fold"

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> PassOutcome:
        folded_operations = 0

        def rewrite_block(block: Block) -> Block:
            nonlocal folded_operations
            rewritten = []
            for operation in block.operations:
                operation = rewrite_regions(operation, rewrite_block)
                if operation.name in FOLDABLE_OPERATIONS and all(
                    operand.id in analysis.constant_values
                    for operand in operation.operands
                ):
                    result_id = operation.results[0].id
                    if result_id in analysis.constant_values:
                        operation = Operation(
                            "arith.constant",
                            results=operation.results,
                            attributes={"value": analysis.constant_values[result_id]},
                            location=operation.location,
                        )
                        folded_operations += 1
                rewritten.append(operation)
            return Block(block.arguments, tuple(rewritten))

        transformed = HybridProgram(
            program.name,
            Region(tuple(rewrite_block(block) for block in program.body.blocks)),
            location=program.location,
        )
        return PassOutcome(
            transformed,
            statistics={"folded_operations": folded_operations},
        )


@dataclass(frozen=True)
class StructuredControlFlowSimplificationPass:
    """Inline constant branches and remove statically empty loops."""

    name: str = "structured_control_flow_simplification"

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> PassOutcome:
        constant_branches_inlined = 0
        empty_loops_removed = 0

        def rewrite_block(
            block: Block,
            inherited: Mapping[ValueId, Value] | None = None,
        ) -> Block:
            nonlocal constant_branches_inlined, empty_loops_removed
            substitutions = dict(inherited or {})
            rewritten: list[Operation] = []
            for original in block.operations:
                operands = tuple(
                    substitute_value(value, substitutions)
                    for value in original.operands
                )
                operation = (
                    original
                    if operands == original.operands
                    else replace(original, operands=operands)
                )
                if self._is_constant_branch(operation, analysis):
                    constant_branches_inlined += 1
                    predicate = bool(analysis.constant_values[operation.operands[0].id])
                    selected = operation.regions[0 if predicate else 1].blocks[0]
                    branch_bindings = dict(substitutions)
                    branch_bindings.update(
                        (argument.id, operand)
                        for argument, operand in zip(
                            selected.arguments, operation.operands[1:]
                        )
                    )
                    selected_block = rewrite_block(selected, branch_bindings)
                    terminator = selected_block.operations[-1]
                    rewritten.extend(selected_block.operations[:-1])
                    substitutions.update(
                        (result.id, yielded)
                        for result, yielded in zip(
                            operation.results, terminator.operands
                        )
                    )
                    continue
                iterations = constant_loop_iterations(operation, analysis)
                if iterations is not None and len(iterations) == 0:
                    empty_loops_removed += 1
                    substitutions.update(
                        (result.id, initial)
                        for result, initial in zip(
                            operation.results, operation.operands[3:]
                        )
                    )
                    continue
                regions = tuple(
                    Region(
                        tuple(
                            rewrite_block(nested, substitutions)
                            for nested in region.blocks
                        )
                    )
                    for region in operation.regions
                )
                rewritten.append(
                    operation
                    if regions == operation.regions
                    else replace(operation, regions=regions)
                )
            return Block(block.arguments, tuple(rewritten))

        transformed = HybridProgram(
            program.name,
            Region(tuple(rewrite_block(block) for block in program.body.blocks)),
            location=program.location,
        )
        return PassOutcome(
            transformed,
            statistics={
                "constant_branches_inlined": constant_branches_inlined,
                "empty_loops_removed": empty_loops_removed,
            },
        )

    @staticmethod
    def _is_constant_branch(
        operation: Operation, analysis: HybridProgramAnalysis
    ) -> bool:
        if operation.name != "scf.if":
            return False
        value = analysis.constant_values.get(operation.operands[0].id)
        return isinstance(value, bool)


@dataclass(frozen=True)
class BoundedLoopUnrollPass:
    """Unroll small constant top-level loops with fresh SSA definitions."""

    max_iterations: int = 8
    max_expanded_operations: int = 256
    name: str = field(default="bounded_loop_unroll", init=False)

    def __post_init__(self) -> None:
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if int(self.max_expanded_operations) <= 0:
            raise ValueError("max_expanded_operations must be positive")

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> PassOutcome:
        statistics = {
            "loops_unrolled": 0,
            "iterations_unrolled": 0,
            "expanded_operations": 0,
            "loops_preserved_dynamic_bounds": 0,
            "loops_deferred_empty": 0,
            "loops_preserved_iteration_budget": 0,
            "loops_preserved_operation_budget": 0,
        }
        remarks: list[str] = []

        def preserve(operation: Operation, reason: str) -> None:
            statistics[reason] += 1
            result_id = operation.results[-1].id
            remarks.append(f"preserved {result_id.scope}:{result_id.index}: {reason}")

        def rewrite_entry(block: Block) -> Block:
            substitutions: dict[ValueId, Value] = {}
            rewritten: list[Operation] = []
            for original in block.operations:
                operation = substitute_operation(original, substitutions)
                if operation.name != "scf.for":
                    rewritten.append(operation)
                    continue
                iterations = constant_loop_iterations(operation, analysis)
                if iterations is None:
                    preserve(operation, "loops_preserved_dynamic_bounds")
                    rewritten.append(operation)
                    continue
                if not iterations:
                    preserve(operation, "loops_deferred_empty")
                    rewritten.append(operation)
                    continue
                if len(iterations) > self.max_iterations:
                    preserve(operation, "loops_preserved_iteration_budget")
                    rewritten.append(operation)
                    continue
                body = operation.regions[0].blocks[0]
                body_operation_count = region_operation_count(operation.regions[0]) - 1
                expansion = len(iterations) * (body_operation_count + 1)
                if (
                    statistics["expanded_operations"] + expansion
                    > self.max_expanded_operations
                ):
                    preserve(operation, "loops_preserved_operation_budget")
                    rewritten.append(operation)
                    continue
                statistics["loops_unrolled"] += 1
                statistics["iterations_unrolled"] += len(iterations)
                statistics["expanded_operations"] += expansion
                carried = operation.operands[3:]
                result_id = operation.results[-1].id
                loop_key = f"unroll_{result_id.scope}_{result_id.index}"
                for ordinal, iteration in enumerate(iterations):
                    suffix = f"{loop_key}_{ordinal}"
                    induction = fresh_value(body.arguments[0], suffix)
                    rewritten.append(
                        Operation(
                            "arith.constant",
                            results=(induction,),
                            attributes={"value": iteration},
                            location=operation.location,
                        )
                    )
                    bindings = {
                        body.arguments[0].id: induction,
                        **{
                            argument.id: value
                            for argument, value in zip(body.arguments[1:], carried)
                        },
                    }
                    cloned = clone_inline_block(body, bindings, suffix=suffix)
                    terminator = cloned.operations[-1]
                    rewritten.extend(cloned.operations[:-1])
                    carried = terminator.operands
                substitutions.update(
                    (result.id, value)
                    for result, value in zip(operation.results, carried)
                )
            return Block(block.arguments, tuple(rewritten))

        transformed = HybridProgram(
            program.name,
            Region(tuple(rewrite_entry(block) for block in program.body.blocks)),
            location=program.location,
        )
        return PassOutcome(transformed, statistics=statistics, remarks=tuple(remarks))


@dataclass(frozen=True)
class DeadConstantEliminationPass:
    """Remove constants whose SSA results have no uses after folding."""

    name: str = "dead_constant_elimination"

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> PassOutcome:
        constants_removed = 0

        def rewrite_block(block: Block) -> Block:
            nonlocal constants_removed
            rewritten = []
            for operation in block.operations:
                operation = rewrite_regions(operation, rewrite_block)
                if operation.name == "arith.constant" and all(
                    analysis.use_counts.get(result.id, 0) == 0
                    for result in operation.results
                ):
                    constants_removed += 1
                    continue
                rewritten.append(operation)
            return Block(block.arguments, tuple(rewritten))

        transformed = HybridProgram(
            program.name,
            Region(tuple(rewrite_block(block) for block in program.body.blocks)),
            location=program.location,
        )
        return PassOutcome(
            transformed,
            statistics={"constants_removed": constants_removed},
        )


__all__ = (
    "BoundedLoopUnrollPass",
    "ConstantFoldPass",
    "DeadConstantEliminationPass",
    "PassOutcome",
    "StructuredControlFlowSimplificationPass",
)
