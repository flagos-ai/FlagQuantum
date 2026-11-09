"""Run python -m examples.algorithms.classical_register_subtraction."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic.addition import negate_ir
from flagquantum.compiler.registers.arithmetic.constant_add_resource_model import (
    source_constant_add_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_negation import negate_classical_value
from flagquantum.simulation.classical_subtraction import (
    subtract_classical_values,
    subtract_from_classical_values,
)


@dataclass(frozen=True)
class NegateWord:
    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = negate_ir((0, 1, 2), (3, 4, 5), (6, 7), (0, 1))
        return circuit.compose(CircuitIR(8, gates))


def main() -> None:
    assert subtract_classical_values(1, 3, 3) == (1, 6)
    assert subtract_from_classical_values(1, 3, 3) == (1, 2)
    assert negate_classical_value(2, 3, signed=True) == -2
    expected_value = negate_classical_value(2, 3)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[2 << 5] = 1
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(NegateWord())
    result = run_dynamic(circuit, shots=16, seed=29, strategy="trajectory")
    expected = torch.zeros_like(initial)
    expected[expected_value << 5] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    model = estimate_component_resources(source_constant_add_cost(3, 1))
    assert counts["t"] + counts["tdg"] == model.gates.total_t_count() == 8
    assert counts["measure"] == model.gates.measurement == 2
    print("Directions: 1-3 mod 8 = 6; 3-1 mod 8 = 2")
    print("circuit.append(op): negate stored word 010 -> 110; all five work wires zero")
    print("Emitted primitive counts:", dict(counts))
    print("Independent AddK model (complement adds three X only):", model)


if __name__ == "__main__":
    main()
