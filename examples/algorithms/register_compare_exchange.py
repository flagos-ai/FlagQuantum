"""Append original offset comparisons, retain histories and undo them exactly.

Run: python -m examples.algorithms.register_compare_exchange
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.parallel_comparison import parallel_compare_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class OffsetComparisons:
    """Bind five one-bit words, two histories and two clean comparison work bits."""

    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = parallel_compare_ir(
            tuple((i,) for i in range(5)), 2, (5, 6), (7, 8), (0,), adjoint=self.inverse
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    v = torch.arange(32, dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for x, amp in enumerate(amps):
        original = [(x >> (4 - i)) & 1 for i in range(5)]
        words = original.copy()
        histories = []
        # Original offset-two block: pair 0/2, then 1/3; trailing word 4 is idle.
        for i, j in ((0, 2), (1, 3)):
            histories.append(int(original[i] > original[j]))
            words[i], words[j] = min(original[i], original[j]), max(
                original[i], original[j]
            )
        out = 0
        for bit in (*words, *histories):
            out = (out << 1) | bit
        state[x << 4] = amp
        expected[out << 2] = amp
    circuit = DynamicCircuit(9, inputs=state, dtype=torch.complex128)
    circuit.append(OffsetComparisons())
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(OffsetComparisons(inverse=True))
    restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=1e-12
    )
    print(
        "Complex amplitudes correct; two logical history bits retained; two quantum work bits zero."
    )
    print(
        "Original adjoint restored all words and cleared histories; trailing word retained."
    )
    print("Measured forward primitives:", dict(counts))
    print(
        "Independent source estimates: exact 7T-CSwap 22 T; default 4T-CSwap 16 T; two measurements; nine total qubits."
    )


if __name__ == "__main__":
    main()
