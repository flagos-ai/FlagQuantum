"""Compare an unsigned register against a constant using circuit.append(op).

Run: python -m examples.algorithms.register_comparison
Expected: target XOR (x<5) preserves complex amplitudes and clears four work
wires. A threshold of 8 flips target with no workspace. No reference dependency.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.comparison.constant_component import (
    less_than_constant_program,
)
from flagquantum.compiler.registers.comparison.less_than import (
    less_than_constant_workspace_size,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ConstantComparison:
    """Bind three source wires, a target and explicit available clean work."""

    threshold: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        size = less_than_constant_workspace_size(3, self.threshold)
        bound = bind_register_program(
            less_than_constant_program(3, self.threshold),
            ((0, 1, 2), (3,)),
            clean=tuple(range(4, 4 + size)),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, bound.circuit.instructions))


def main() -> None:
    """Check every input/target amplitude, not only a basis truth table."""
    data = torch.arange(16, dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(256, dtype=torch.complex128)
    state[data << 4] = amplitudes
    for threshold in (5, 8):
        operation = ConstantComparison(threshold)
        circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
        circuit.append(operation)
        mapped = ((data >> 1) << 1) | ((data & 1) ^ ((data >> 1) < threshold))
        expected = torch.zeros_like(state)
        expected[mapped << 4] = amplitudes
        result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(32, -1),
            expected.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        circuit.append(operation)
        restored = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            restored.final_states.reshape(32, -1),
            state.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            f"target XOR (x<{threshold}): all complex amplitudes correct; work is zero; twice restores input"
        )
        print("Required clean work:", less_than_constant_workspace_size(3, threshold))
        print(
            "Measured gates:",
            dict(counts),
        )


if __name__ == "__main__":
    main()
