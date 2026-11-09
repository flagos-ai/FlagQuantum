"""Append the pinned difference-sign circuit on its no-overflow signed domain.

Run: python -m examples.algorithms.register_signed_difference_sign
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.comparison.linear_comparison import (
    linear_greater_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class DifferenceSign:
    """Bind two three-bit words, an arbitrary XOR target and two clean carries."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = linear_greater_ir((0, 1, 2), (3, 4, 5), 6, (7, 8), (0, 1), signed=True)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    # Values -1, 0, 1 cannot overflow a three-bit b-a subtraction.
    values = (-1, 0, 1)
    labels = [(a, b, z) for a in values for b in values for z in (0, 1)]
    v = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (a, b, z), amp in zip(labels, amps, strict=True):
        pair = ((a % 8) << 3) | (b % 8)
        state[((pair << 1) | z) << 2] = amp
        expected[((pair << 1) | (z ^ int(a > b))) << 2] = amp
    circuit = DynamicCircuit(9, inputs=state, dtype=torch.complex128)
    circuit.append(DifferenceSign())
    result = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(DifferenceSign())
    restored = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(16, -1), state.expand(16, -1), rtol=0, atol=1e-12
    )
    print("Complex amplitudes correct on the no-overflow domain; two work bits zero.")
    print("Second append restored both words and the arbitrary target.")
    print("Measured forward primitives:", dict(counts))
    print("Independent source estimates: 8 T; 2 measurements; 9 total qubits.")
    print(
        "Source exception: overflow and width-one cases do not implement full signed ordering."
    )


if __name__ == "__main__":
    main()
