"""Aggregate traced calls and their exclusive emitted instruction counts."""

from collections import Counter

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import CircuitIR
from .trace import ComponentCall, ComponentTrace


def summarize_components(
    ir: CircuitIR, trace: ComponentTrace
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str, bool]]]:
    """Return edge multiplicities and counts by component, opcode and condition.

    Instructions outside recorded roots belong to the circuit. A parent
    owns the instructions in its interval that are outside child intervals.
    """
    edges: Counter[tuple[str, str]] = Counter()
    gates: Counter[tuple[str, str, bool]] = Counter()

    def add_gates(name: str, start: int, stop: int) -> None:
        gates.update(
            (name, op.name, bool(instruction_condition_clauses(op)))
            for op in ir.instructions[start:stop]
        )

    def visit(
        parent: str, start: int, stop: int, calls: tuple[ComponentCall, ...]
    ) -> None:
        cursor = start
        for call in calls:
            if not cursor <= call.start <= call.stop <= stop:
                raise ValueError("Component intervals must be ordered and nested")
            add_gates(parent, cursor, call.start)
            edges[parent, call.name] += 1
            visit(call.name, call.start, call.stop, call.children)
            cursor = call.stop
        add_gates(parent, cursor, stop)

    visit("Circuit", 0, len(ir.instructions), tuple(trace.calls))
    return edges, gates
