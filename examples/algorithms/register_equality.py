"""Run register and constant equality through circuit.append(op).

Run: python -m examples.algorithms.register_equality
Expected: arbitrary target XOR equality, every complex amplitude preserved,
clean work, and double comparison restoring the input. No reference dependency.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.equality import equality_constant_ir, equality_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class Equality:
    """Bind three-bit inputs and two explicit clean work wires."""

    constant: int | None = None

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        if self.constant is None:
            gates = equality_ir((0, 1, 2), (3, 4, 5), 6, (7, 8), (0, 1))
        else:
            gates = equality_constant_ir((0, 1, 2), self.constant, 3, (4, 5), (0, 1))
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    """Compare all inputs and both target values, not only a zero target."""
    for constant in (None, 5):
        data_bits = 7 if constant is None else 4
        n_qubits = data_bits + 2
        data = torch.arange(1 << data_bits, dtype=torch.int64)
        values, target = data >> 1, data & 1
        predicate = (
            ((values >> 3) == (values & 7)) if constant is None else values == constant
        )
        mapped = (values << 1) | (target ^ predicate)
        coeff = data.to(torch.float64)
        amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
        amplitudes /= amplitudes.norm()
        state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
        state[data << 2] = amplitudes
        expected = torch.zeros_like(state)
        expected[mapped << 2] = amplitudes
        operation = Equality(constant)
        circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
            operation
        )
        result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(32, -1),
            expected.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        restored = run_dynamic(
            circuit.append(operation), shots=32, seed=41, strategy="trajectory"
        )
        torch.testing.assert_close(
            restored.final_states.reshape(32, -1),
            state.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        label = "x == y" if constant is None else f"x == {constant}"
        print(
            f"{label}: all complex amplitudes correct, both work wires zero, twice restores input"
        )
        print("Measured gates for one comparison:", dict(counts))


if __name__ == "__main__":
    main()
