"""Run python -m examples.algorithms.grouped_controlled_constant_addition."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.multi_controlled_constant_addition import (
    grouped_controlled_add_constant_ir,
    source_grouped_controlled_constant_add_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class GroupedConstantAddition:
    value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        registers = (ControlRegister(2, (-1,), "twos_complement"),)
        gates = grouped_controlled_add_constant_ir(
            ((0, 1),), registers, (2, 3), self.value, (4,), (5, 6), (7,), (0,), (1,)
        )
        return circuit.compose(CircuitIR(8, gates))


def main() -> None:
    state = torch.zeros(256, dtype=torch.complex128)
    state[(3 << 6) | (1 << 4)] = 1
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(GroupedConstantAddition(-1))
    result = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    expected = torch.zeros_like(state)
    expected[3 << 6] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    registers = (ControlRegister(2, (-1,), "twos_complement"),)
    model = estimate_component_resources(
        source_grouped_controlled_constant_add_cost(2, -1, registers)
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == model.gates.total_t_count()
    assert counts["measure"] == model.gates.measurement
    print("Signed control -1: target 1-1=0; all four work bits clear")
    print("Logical source model:", model)
    print("Emitted primitive counts:", dict(counts))


if __name__ == "__main__":
    main()
