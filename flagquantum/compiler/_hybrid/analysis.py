"""Verified analyses for the private hybrid program IR."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .model import HybridProgram, Operation, Region, ValueId
from .verifier import verify_program

FOLDABLE_OPERATIONS = frozenset(
    {
        "arith.add",
        "arith.rem",
        "arith.not",
        "arith.and",
        "arith.or",
        "arith.cmp",
    }
)


def operations(program: HybridProgram) -> tuple[Operation, ...]:
    """Return operations in deterministic region preorder."""

    items: list[Operation] = []

    def visit_region(region: Region) -> None:
        for block in region.blocks:
            for operation in block.operations:
                items.append(operation)
                for nested in operation.regions:
                    visit_region(nested)

    visit_region(program.body)
    return tuple(items)


def fold_value(operation: Operation, operands: tuple[Any, ...]) -> Any:
    """Evaluate one operation already proven to have constant operands."""

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


@dataclass(frozen=True)
class HybridProgramAnalysis:
    """Constant, use-count, and size facts for one verified program."""

    constant_values: Mapping[ValueId, Any]
    use_counts: Mapping[ValueId, int]
    operation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "constant_values", MappingProxyType(dict(self.constant_values))
        )
        object.__setattr__(self, "use_counts", MappingProxyType(dict(self.use_counts)))


def analyze_program(program: HybridProgram) -> HybridProgramAnalysis:
    """Compute deterministic facts after verifying the complete program."""

    verify_program(program)
    program_operations = operations(program)
    constants: dict[ValueId, Any] = {}
    uses: Counter[ValueId] = Counter()
    for operation in program_operations:
        uses.update(value.id for value in operation.operands)
        if operation.name == "arith.constant":
            constants[operation.results[0].id] = operation.attributes["value"]
        elif operation.name in FOLDABLE_OPERATIONS and all(
            operand.id in constants for operand in operation.operands
        ):
            operands = tuple(constants[operand.id] for operand in operation.operands)
            try:
                constants[operation.results[0].id] = fold_value(operation, operands)
            except (ArithmeticError, TypeError, ValueError):
                pass
    return HybridProgramAnalysis(constants, uses, len(program_operations))


def constant_loop_iterations(
    operation: Operation, analysis: HybridProgramAnalysis
) -> range | None:
    """Return a constant iteration range, or ``None`` when not established."""

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


def direct_constant_loop_counts(
    program: HybridProgram, analysis: HybridProgramAnalysis
) -> Mapping[ValueId, int]:
    """Return nonempty constant-loop counts directly in the program entry block."""

    return MappingProxyType(
        {
            operation.results[-1].id: len(iterations)
            for block in program.body.blocks
            for operation in block.operations
            if (iterations := constant_loop_iterations(operation, analysis)) is not None
            and len(iterations) > 0
        }
    )


__all__ = (
    "FOLDABLE_OPERATIONS",
    "HybridProgramAnalysis",
    "analyze_program",
    "constant_loop_iterations",
    "direct_constant_loop_counts",
    "fold_value",
    "operations",
)
