"""Compare a component cost model with one generated circuit's IR."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import CircuitIR
from .resource_model import (
    ComponentCost,
    ModeledResources,
    estimate_component_resources,
)


@dataclass(frozen=True)
class EmittedInstructionCounts:
    """Syntactic counts before branch selection or execution."""

    declared_qubits: int
    instruction_qubits: int
    unconditional: dict[str, int]
    conditional: dict[str, int]
    observable_requests: int
    measurement_requests: int

    @property
    def t_count(self) -> int:
        """Number of emitted T and T-dagger instructions across both groups."""
        return sum(
            counts.get(opcode, 0)
            for counts in (self.unconditional, self.conditional)
            for opcode in ("t", "tdg")
        )


def count_emitted_instructions(ir: CircuitIR) -> EmittedInstructionCounts:
    """Count each IR instruction once, regardless of its condition or clauses."""
    unconditional: Counter[str] = Counter()
    conditional: Counter[str] = Counter()
    used: set[int] = set()
    for instruction in ir.instructions:
        counts = (
            conditional if instruction_condition_clauses(instruction) else unconditional
        )
        counts[instruction.name] += 1
        used.update(instruction.wires)
    return EmittedInstructionCounts(
        ir.n_wires,
        len(used),
        dict(sorted(unconditional.items())),
        dict(sorted(conditional.items())),
        len(ir.observables),
        len(ir.measurements),
    )


@dataclass(frozen=True)
class ComponentResourceComparison:
    """Keep logical model and generated instruction counts side by side."""

    component: str
    model: ModeledResources
    emitted: EmittedInstructionCounts

    def to_dict(self) -> dict[str, Any]:
        """Return the two bases explicitly in a JSON-compatible report."""
        return {
            "component": self.component,
            "component_model": {
                **asdict(self.model),
                "default_t_equivalents": self.model.gates.total_t_count(),
            },
            "emitted_ir": {**asdict(self.emitted), "t_count": self.emitted.t_count},
        }


def compare_component_resources(
    component: ComponentCost, ir: CircuitIR
) -> ComponentResourceComparison:
    """Combine model estimates and IR counts for a caller-paired component and circuit."""
    return ComponentResourceComparison(
        component.name,
        estimate_component_resources(component),
        count_emitted_instructions(ir),
    )
