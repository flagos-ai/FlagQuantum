"""Append coherent conditional addition/subtraction and its original inverse.

Run: python -m examples.algorithms.register_controlled_add_or_subtract
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic.addition import (
    measurement_add_or_subtract_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ConditionalArithmetic:
    """Bind coherent control, two three-bit words and two clean carry wires."""

    add_on: bool = True

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = measurement_add_or_subtract_ir(
            0,
            (1, 2, 3),
            (4, 5, 6),
            (7, 8),
            (0, 1),
            add_when_ctrl_is_on=self.add_on,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    v = torch.arange(128, dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for raw, amp in enumerate(amps):
        control, a, b = raw >> 6, (raw >> 3) & 7, raw & 7
        word = (b + a if control else b - a) % 8
        state[raw << 2] = amp
        expected[(((control << 3) | a) << 3 | word) << 2] = amp
    circuit = DynamicCircuit(9, inputs=state, dtype=torch.complex128)
    circuit.append(ConditionalArithmetic())
    result = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1),
        expected.expand(16, -1),
        rtol=0,
        atol=1e-12,
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(ConditionalArithmetic(add_on=False))
    restored = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(16, -1),
        state.expand(16, -1),
        rtol=0,
        atol=1e-12,
    )
    print(
        "Complex amplitudes correct for both coherent control sectors; two work bits zero."
    )
    print("Original opposite-polarity adjoint restored both words and the control.")
    print("Measured forward primitives:", dict(counts))
    print("Independent source estimates: 8 T; 2 measurements; 9 total qubits.")


if __name__ == "__main__":
    main()
