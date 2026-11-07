"""Run python -m examples.algorithms.grouped_constant_xor."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.grouped_constant_xor import (
    grouped_controlled_xor_constant_ir,
    source_grouped_constant_xor_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class GroupedConstantXor:
    value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = grouped_controlled_xor_constant_ir(
            ((0, 1),), (ControlRegister(2, (2,)),), (2, 3), self.value, (4,), (0,)
        )
        return circuit.compose(CircuitIR(5, gates))


def main() -> None:
    state = torch.zeros(32, dtype=torch.complex128)
    state[(2 << 3) | (1 << 1)] = 1
    circuit = DynamicCircuit(5, inputs=state, dtype=torch.complex128)
    circuit.append(GroupedConstantXor(3))
    result = run_dynamic(circuit, shots=16, seed=37, strategy="trajectory")
    expected = torch.zeros_like(state)
    expected[(2 << 3) | (2 << 1)] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    model = estimate_component_resources(
        source_grouped_constant_xor_cost(2, 3, (ControlRegister(2, (2,)),))
    )
    assert counts["t"] + counts["tdg"] == model.gates.total_t_count()
    assert counts["measure"] == model.gates.measurement
    print("Control word 2: 1 XOR 3 = 2; predicate work returned to zero")
    print("Declared logical model:", model)
    print("Emitted primitive counts:", dict(counts))


if __name__ == "__main__":
    main()
