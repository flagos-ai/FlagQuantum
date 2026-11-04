"""Shared Clifford+T accounting for the optional parity suite."""

from collections import Counter
from collections.abc import Iterable
from typing import Any


def clifford_t_counts(cirq: Any, operations: Iterable[Any]) -> Counter[str]:
    """Count a decomposed circuit with a common Clifford+T convention."""
    counts = Counter()
    for operation in operations:
        gate = operation.gate
        if isinstance(operation, cirq.ClassicallyControlledOperation):
            assert cirq.has_stabilizer_effect(operation.without_classical_controls())
            category = "feedback"
        elif cirq.is_measurement(operation):
            category = "measurement"
        elif isinstance(gate, cirq.ResetChannel):
            category = "reset"
        elif isinstance(gate, cirq.GlobalPhaseGate):
            continue
        elif cirq.has_stabilizer_effect(operation):
            category = "clifford"
        else:
            assert isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate)), (
                operation
            )
            quarters = 4 * float(gate.exponent)
            assert abs(quarters - round(quarters)) < 1e-9, operation
            category = "t" if round(quarters) % 2 else "clifford"
        counts[category] += 1
    return counts
