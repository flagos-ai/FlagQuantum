"""Run python -m examples.algorithms.classical_register_addition."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic_resource_model import (
    source_controlled_add_cost,
)
from flagquantum.compiler.registers.controlled_addition import controlled_add_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_addition import (
    add_constant_classical_value,
    out_of_place_sum_classical_values,
    register_add_classical_values,
)
from flagquantum.simulation.classical_controlled_addition import (
    controlled_add_classical_values,
)


@dataclass(frozen=True)
class ControlledWordAddition:
    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = controlled_add_ir(0, (1, 2), (3, 4), (5, 6), (7,), (0, 1), (2,))
        return circuit.compose(CircuitIR(8, gates))


def main() -> None:
    assert register_add_classical_values(3, 2, 2) == (3, 1)
    assert add_constant_classical_value(2, -3, 3) == 7
    assert out_of_place_sum_classical_values(3, 2, 2)["c"] == 5
    control, source, target = controlled_add_classical_values(1, 3, 2, 2)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[(1 << 7) | (3 << 5) | (2 << 3)] = 1
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(ControlledWordAddition())
    result = run_dynamic(circuit, shots=16, seed=17, strategy="trajectory")
    expected = torch.zeros_like(initial)
    expected[(int(control) << 7) | (int(source) << 5) | (int(target) << 3)] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 12
    assert counts["measure"] == 3
    print("Classical model and circuit.append(op): control 1, source 3, target 2 -> 1")
    print("Source/control preserved; all three work wires zero")
    print("Emitted primitive counts:", dict(counts))
    model = estimate_component_resources(source_controlled_add_cost(2, 2))
    assert model.gates.total_t_count() == 12 and model.gates.measurement == 3
    print("Independent logical resource model:", model)


if __name__ == "__main__":
    main()
