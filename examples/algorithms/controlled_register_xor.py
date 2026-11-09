"""Run python -m examples.algorithms.controlled_register_xor."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.bits.controlled_register_xor import (
    grouped_controlled_xor_ir,
    source_grouped_xor_cost,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledRegisterXor:
    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = grouped_controlled_xor_ir(
            ((0, 1),), (ControlRegister(2, (2,)),), (2, 3), (4, 5), (6,), (0,)
        )
        return circuit.compose(CircuitIR(7, gates))


def main() -> None:
    state = torch.zeros(128, dtype=torch.complex128)
    state[(2 << 5) | (3 << 3) | (1 << 1)] = 1
    circuit = DynamicCircuit(7, inputs=state, dtype=torch.complex128)
    circuit.append(ControlledRegisterXor())
    result = run_dynamic(circuit, shots=16, seed=37, strategy="trajectory")
    expected = torch.zeros_like(state)
    expected[(2 << 5) | (3 << 3) | (2 << 1)] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    report = estimate_component_resources(
        source_grouped_xor_cost(2, (ControlRegister(2, (2,)),))
    )
    assert counts["ccx"] == report.gates.toffoli == 2
    assert counts["t"] + counts["tdg"] == 4
    assert counts["measure"] == report.gates.measurement == 1
    print("Control word 2: target 1 XOR source 3 = 2; source preserved, work zero")
    print("Emitted Clifford+Toffoli instructions:", dict(counts))
    print("Logical resource model:", report)
    print("Model T equivalents (4 T per atomic Toffoli):", report.gates.total_t_count())


if __name__ == "__main__":
    main()
