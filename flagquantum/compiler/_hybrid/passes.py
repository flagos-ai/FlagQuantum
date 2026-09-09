"""Verified optimization passes for the private hybrid program IR."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .model import Block, HybridProgram, Operation, Region, Value, ValueId
from .verifier import verify_program


def _operations(program: HybridProgram) -> tuple[Operation, ...]:
    items: list[Operation] = []

    def visit_region(region: Region) -> None:
        for block in region.blocks:
            for operation in block.operations:
                items.append(operation)
                for nested in operation.regions:
                    visit_region(nested)

    visit_region(program.body)
    return tuple(items)


def _fold_value(operation: Operation, operands: tuple[Any, ...]) -> Any:
    name = operation.name
    if name == "arith.add":
        return operands[0] + operands[1]
    if name == "arith.rem":
        return operands[0] % operands[1]
    if name == "arith.not":
        return not operands[0]
    if name == "arith.and":
        return operands[0] and operands[1]
    if name == "arith.or":
        return operands[0] or operands[1]
    if name == "arith.cmp":
        predicate = operation.attributes["predicate"]
        return {
            "eq": lambda: operands[0] == operands[1],
            "ne": lambda: operands[0] != operands[1],
            "lt": lambda: operands[0] < operands[1],
            "le": lambda: operands[0] <= operands[1],
            "gt": lambda: operands[0] > operands[1],
            "ge": lambda: operands[0] >= operands[1],
        }[predicate]()
    raise KeyError(name)


_FOLDABLE = frozenset(
    {
        "arith.add",
        "arith.rem",
        "arith.not",
        "arith.and",
        "arith.or",
        "arith.cmp",
    }
)


@dataclass(frozen=True)
class HybridProgramAnalysis:
    """Facts consumed by the first two concrete optimization passes."""

    constant_values: Mapping[ValueId, Any]
    use_counts: Mapping[ValueId, int]
    operation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "constant_values", MappingProxyType(dict(self.constant_values))
        )
        object.__setattr__(self, "use_counts", MappingProxyType(dict(self.use_counts)))


def analyze_program(program: HybridProgram) -> HybridProgramAnalysis:
    """Compute deterministic constant and use-count facts for a verified program."""

    verify_program(program)
    operations = _operations(program)
    constants: dict[ValueId, Any] = {}
    uses: Counter[ValueId] = Counter()
    for operation in operations:
        uses.update(value.id for value in operation.operands)
        if operation.name == "arith.constant":
            constants[operation.results[0].id] = operation.attributes["value"]
        elif operation.name in _FOLDABLE and all(
            operand.id in constants for operand in operation.operands
        ):
            operands = tuple(constants[operand.id] for operand in operation.operands)
            try:
                constants[operation.results[0].id] = _fold_value(operation, operands)
            except (ArithmeticError, TypeError, ValueError):
                pass
    return HybridProgramAnalysis(constants, uses, len(operations))


class HybridProgramPass(Protocol):
    """One private transformation over a verified HybridProgram."""

    name: str

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> HybridProgram: ...


def _rewrite_regions(
    operation: Operation,
    rewrite_block: Callable[[Block], Block],
) -> Operation:
    regions = tuple(
        Region(tuple(rewrite_block(block) for block in region.blocks))
        for region in operation.regions
    )
    return (
        operation
        if regions == operation.regions
        else replace(operation, regions=regions)
    )


@dataclass(frozen=True)
class ConstantFoldPass:
    """Replace constant-only arithmetic with typed constants."""

    name: str = "constant_fold"

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> HybridProgram:
        def rewrite_block(block: Block) -> Block:
            rewritten = []
            for operation in block.operations:
                operation = _rewrite_regions(operation, rewrite_block)
                if operation.name in _FOLDABLE and all(
                    operand.id in analysis.constant_values
                    for operand in operation.operands
                ):
                    value = analysis.constant_values.get(operation.results[0].id)
                    if operation.results[0].id in analysis.constant_values:
                        operation = Operation(
                            "arith.constant",
                            results=operation.results,
                            attributes={"value": value},
                            location=operation.location,
                        )
                rewritten.append(operation)
            return Block(block.arguments, tuple(rewritten))

        return HybridProgram(
            program.name,
            Region(tuple(rewrite_block(block) for block in program.body.blocks)),
            location=program.location,
        )


def _substitute_value(value: Value, substitutions: Mapping[ValueId, Value]) -> Value:
    visited: set[ValueId] = set()
    current = value
    while current.id in substitutions:
        if current.id in visited:
            raise ValueError("hybrid SSA substitution contains a cycle")
        visited.add(current.id)
        current = substitutions[current.id]
    return current


def _substitute_operation(
    operation: Operation, substitutions: Mapping[ValueId, Value]
) -> Operation:
    operands = tuple(
        _substitute_value(value, substitutions) for value in operation.operands
    )
    regions = tuple(
        Region(
            tuple(
                Block(
                    block.arguments,
                    tuple(
                        _substitute_operation(nested, substitutions)
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


def _constant_loop_iterations(
    operation: Operation, analysis: HybridProgramAnalysis
) -> range | None:
    if operation.name != "scf.for":
        return None
    bounds = tuple(
        analysis.constant_values.get(operand.id) for operand in operation.operands[:3]
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in bounds):
        return None
    lower, upper, step = bounds
    if step == 0:
        return None
    return range(lower, upper, step)


@dataclass(frozen=True)
class StructuredControlFlowSimplificationPass:
    """Inline constant branches and remove statically empty loops."""

    name: str = "structured_control_flow_simplification"

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> HybridProgram:
        def rewrite_block(
            block: Block,
            inherited: Mapping[ValueId, Value] | None = None,
        ) -> Block:
            substitutions = dict(inherited or {})
            rewritten: list[Operation] = []
            for original in block.operations:
                operands = tuple(
                    _substitute_value(value, substitutions)
                    for value in original.operands
                )
                operation = (
                    original
                    if operands == original.operands
                    else replace(original, operands=operands)
                )
                if self._is_constant_branch(operation, analysis):
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
                if self._is_empty_loop(operation, analysis):
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

        return HybridProgram(
            program.name,
            Region(tuple(rewrite_block(block) for block in program.body.blocks)),
            location=program.location,
        )

    @staticmethod
    def _is_constant_branch(
        operation: Operation, analysis: HybridProgramAnalysis
    ) -> bool:
        if operation.name != "scf.if":
            return False
        value = analysis.constant_values.get(operation.operands[0].id)
        return isinstance(value, bool)

    @staticmethod
    def _is_empty_loop(operation: Operation, analysis: HybridProgramAnalysis) -> bool:
        iterations = _constant_loop_iterations(operation, analysis)
        return iterations is not None and len(iterations) == 0


def _fresh_value(value: Value, suffix: str) -> Value:
    return Value(
        ValueId(f"{value.id.scope}.{suffix}", value.id.index),
        value.type,
    )


def _clone_inline_block(
    block: Block,
    bindings: Mapping[ValueId, Value],
    *,
    suffix: str,
) -> Block:
    def clone_block(
        nested: Block,
        inherited: Mapping[ValueId, Value],
        argument_bindings: Mapping[ValueId, Value] | None = None,
    ) -> Block:
        substitutions = dict(inherited)
        if argument_bindings is None:
            arguments = tuple(_fresh_value(value, suffix) for value in nested.arguments)
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
                _substitute_value(value, substitutions) for value in operation.operands
            )
            results = tuple(_fresh_value(value, suffix) for value in operation.results)
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


def _region_operation_count(region: Region) -> int:
    return sum(
        1 + sum(_region_operation_count(nested) for nested in operation.regions)
        for block in region.blocks
        for operation in block.operations
    )


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
    ) -> HybridProgram:
        expanded_operations = 0

        def rewrite_entry(block: Block) -> Block:
            nonlocal expanded_operations
            substitutions: dict[ValueId, Value] = {}
            rewritten: list[Operation] = []
            for original in block.operations:
                operation = _substitute_operation(original, substitutions)
                iterations = _constant_loop_iterations(operation, analysis)
                if iterations is None or not 0 < len(iterations) <= self.max_iterations:
                    rewritten.append(operation)
                    continue
                body = operation.regions[0].blocks[0]
                body_operation_count = _region_operation_count(operation.regions[0]) - 1
                expansion = len(iterations) * (body_operation_count + 1)
                if expanded_operations + expansion > self.max_expanded_operations:
                    rewritten.append(operation)
                    continue
                expanded_operations += expansion
                carried = operation.operands[3:]
                loop_key = f"unroll_{operation.results[-1].id.scope}_{operation.results[-1].id.index}"
                for ordinal, iteration in enumerate(iterations):
                    suffix = f"{loop_key}_{ordinal}"
                    induction = _fresh_value(body.arguments[0], suffix)
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
                    cloned = _clone_inline_block(body, bindings, suffix=suffix)
                    terminator = cloned.operations[-1]
                    rewritten.extend(cloned.operations[:-1])
                    carried = terminator.operands
                substitutions.update(
                    (result.id, value)
                    for result, value in zip(operation.results, carried)
                )
            return Block(block.arguments, tuple(rewritten))

        return HybridProgram(
            program.name,
            Region(tuple(rewrite_entry(block) for block in program.body.blocks)),
            location=program.location,
        )


@dataclass(frozen=True)
class DeadConstantEliminationPass:
    """Remove constants whose SSA results have no uses after folding."""

    name: str = "dead_constant_elimination"

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> HybridProgram:
        def rewrite_block(block: Block) -> Block:
            rewritten = []
            for operation in block.operations:
                operation = _rewrite_regions(operation, rewrite_block)
                if operation.name == "arith.constant" and all(
                    analysis.use_counts.get(result.id, 0) == 0
                    for result in operation.results
                ):
                    continue
                rewritten.append(operation)
            return Block(block.arguments, tuple(rewritten))

        return HybridProgram(
            program.name,
            Region(tuple(rewrite_block(block) for block in program.body.blocks)),
            location=program.location,
        )


@dataclass(frozen=True)
class PassRecord:
    """Auditable identity and size transition for one verified pass."""

    name: str
    input_identity: str
    output_identity: str
    input_operation_count: int
    output_operation_count: int
    preexpanded_iterations: int = 0

    @property
    def changed(self) -> bool:
        return self.input_identity != self.output_identity


@dataclass(frozen=True)
class HybridOptimizationResult:
    """Verified optimized program plus deterministic pass evidence."""

    program: HybridProgram
    source_identity: str
    optimized_identity: str
    records: tuple[PassRecord, ...]

    @property
    def changed(self) -> bool:
        return self.source_identity != self.optimized_identity

    @property
    def preexpanded_iterations(self) -> int:
        return sum(record.preexpanded_iterations for record in self.records)


DEFAULT_HYBRID_PASSES: tuple[HybridProgramPass, ...] = (
    BoundedLoopUnrollPass(),
    ConstantFoldPass(),
    StructuredControlFlowSimplificationPass(),
    DeadConstantEliminationPass(),
)


def _direct_constant_loop_counts(
    program: HybridProgram, analysis: HybridProgramAnalysis
) -> Mapping[ValueId, int]:
    return {
        operation.results[-1].id: len(iterations)
        for block in program.body.blocks
        for operation in block.operations
        if (iterations := _constant_loop_iterations(operation, analysis)) is not None
        and len(iterations) > 0
    }


def run_pass_pipeline(
    program: HybridProgram,
    passes: Sequence[HybridProgramPass] = DEFAULT_HYBRID_PASSES,
) -> HybridOptimizationResult:
    """Run a bounded sequence of transformations with verification per step."""

    selected = tuple(passes)
    if len(selected) > 32:
        raise ValueError("hybrid pass pipeline exceeds the 32-pass limit")
    current = program
    current_analysis = analyze_program(current)
    records = []
    for item in selected:
        name = str(getattr(item, "name", "")).strip()
        if not name or not callable(getattr(item, "run", None)):
            raise TypeError("hybrid passes require a name and run method")
        transformed = item.run(current, current_analysis)
        if not isinstance(transformed, HybridProgram):
            raise TypeError(f"hybrid pass {name!r} must return HybridProgram")
        after = analyze_program(transformed)
        before_loops = _direct_constant_loop_counts(current, current_analysis)
        after_loop_ids = set(_direct_constant_loop_counts(transformed, after))
        preexpanded_iterations = sum(
            count
            for loop_id, count in before_loops.items()
            if loop_id not in after_loop_ids
        )
        records.append(
            PassRecord(
                name,
                current.semantic_identity,
                transformed.semantic_identity,
                current_analysis.operation_count,
                after.operation_count,
                preexpanded_iterations,
            )
        )
        current = transformed
        current_analysis = after
    return HybridOptimizationResult(
        current,
        source_identity=program.semantic_identity,
        optimized_identity=current.semantic_identity,
        records=tuple(records),
    )


__all__ = (
    "BoundedLoopUnrollPass",
    "DEFAULT_HYBRID_PASSES",
    "ConstantFoldPass",
    "DeadConstantEliminationPass",
    "HybridOptimizationResult",
    "HybridProgramAnalysis",
    "HybridProgramPass",
    "PassRecord",
    "StructuredControlFlowSimplificationPass",
    "analyze_program",
    "run_pass_pipeline",
)
