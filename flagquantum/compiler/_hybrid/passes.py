"""Verified optimization passes for the private hybrid program IR."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .model import Block, HybridProgram, Operation, Region, ValueId
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


DEFAULT_HYBRID_PASSES: tuple[HybridProgramPass, ...] = (
    ConstantFoldPass(),
    DeadConstantEliminationPass(),
)


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
        records.append(
            PassRecord(
                name,
                current.semantic_identity,
                transformed.semantic_identity,
                current_analysis.operation_count,
                after.operation_count,
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
    "DEFAULT_HYBRID_PASSES",
    "ConstantFoldPass",
    "DeadConstantEliminationPass",
    "HybridOptimizationResult",
    "HybridProgramAnalysis",
    "HybridProgramPass",
    "PassRecord",
    "analyze_program",
    "run_pass_pipeline",
)
