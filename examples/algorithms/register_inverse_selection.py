"""Append original inverse ordering/exchange layers with coherent logical flags.

Run: python -m examples.algorithms.register_inverse_selection
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.inversion.inverse_exchange import (
    inverse_exchange_ir,
)
from flagquantum.compiler.registers.modular.inversion.inverse_ordering import (
    inverse_ordering_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class InverseSelection:
    """Bind four one-bit words, four logical flags and five clean work wires."""

    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        ordering = inverse_ordering_ir(
            (0,), (1,), 4, 5, 6, 7, (8, 9, 10, 11, 12), (0, 1), adjoint=self.adjoint
        )
        exchange = inverse_exchange_ir(5, (0,), (1,), (2,), (3,), adjoint=self.adjoint)
        gates = (*exchange, *ordering) if self.adjoint else (*ordering, *exchange)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    v = torch.arange(256, dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(8192, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for raw, amp in enumerate(amps):
        u, value, r, s, branch, parity, marker, active = tuple(
            raw >> i & 1 for i in reversed(range(8))
        )
        condition = int(u > value) & (branch ^ 1) & active
        parity ^= condition
        marker ^= condition
        if parity:
            u, value, r, s = value, u, s, r
        out = 0
        for bit in (u, value, r, s, branch, parity, marker, active):
            out = (out << 1) | bit
        state[raw << 5] = amp
        expected[out << 5] = amp
    circuit = DynamicCircuit(13, inputs=state, dtype=torch.complex128)
    circuit.append(InverseSelection())
    result = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(InverseSelection(adjoint=True))
    restored = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(16, -1), state.expand(16, -1), rtol=0, atol=1e-12
    )
    print("Complex amplitudes and logical flags correct; five work bits zero.")
    print(
        "Original reverse layers restored all words and flags; this is not a full inverse."
    )
    print("Measured forward primitives:", dict(counts))
    print(
        "Independent source estimates: default CSwap 20 T; exact seven-T CSwap 26 T; three measurements; thirteen peak qubits."
    )


if __name__ == "__main__":
    main()
