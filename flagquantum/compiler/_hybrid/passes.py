"""Verified pass pipeline for the private hybrid program IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .analysis import (
    HybridProgramAnalysis,
    analyze_program,
    direct_constant_loop_counts,
)
from .model import HybridProgram
from .transforms import (
    BoundedLoopUnrollPass,
    ConstantFoldPass,
    DeadConstantEliminationPass,
    PassOutcome,
    StructuredControlFlowSimplificationPass,
)


class HybridProgramPass(Protocol):
    """One private transformation over a verified HybridProgram."""

    name: str

    def run(
        self, program: HybridProgram, analysis: HybridProgramAnalysis
    ) -> HybridProgram | PassOutcome: ...


@dataclass(frozen=True)
class PassRecord:
    """Auditable identity, size, statistics, and remarks for one pass."""

    name: str
    input_identity: str
    output_identity: str
    input_operation_count: int
    output_operation_count: int
    preexpanded_iterations: int = 0
    statistics: Mapping[str, int] = field(default_factory=dict)
    remarks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistics", MappingProxyType(dict(self.statistics)))
        object.__setattr__(self, "remarks", tuple(self.remarks))

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
    ConstantFoldPass(),
    StructuredControlFlowSimplificationPass(),
    BoundedLoopUnrollPass(),
    ConstantFoldPass(),
    StructuredControlFlowSimplificationPass(),
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
        raw_outcome = item.run(current, current_analysis)
        if isinstance(raw_outcome, HybridProgram):
            outcome = PassOutcome(raw_outcome)
        elif isinstance(raw_outcome, PassOutcome):
            outcome = raw_outcome
        else:
            raise TypeError(
                f"hybrid pass {name!r} must return HybridProgram or PassOutcome"
            )
        transformed = outcome.program
        after = analyze_program(transformed)
        before_loops = direct_constant_loop_counts(current, current_analysis)
        after_loop_ids = set(direct_constant_loop_counts(transformed, after))
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
                outcome.statistics,
                outcome.remarks,
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
    "PassOutcome",
    "PassRecord",
    "StructuredControlFlowSimplificationPass",
    "analyze_program",
    "run_pass_pipeline",
)
