"""Append original modular-inverse flag layers and restore their logical data.

Run: python -m examples.algorithms.register_inverse_flags
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.inversion.inverse_parity import (
    inverse_parity_ir,
)
from flagquantum.compiler.registers.modular.inversion.inverse_termination import (
    inverse_termination_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class InverseFlags:
    """Bind two two-bit words, five logical flags and two clean shared work bits."""

    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        termination = inverse_termination_ir(
            (2, 3), 4, 5, 6, (9, 10), (0, 1), adjoint=self.adjoint
        )
        parity = inverse_parity_ir(
            (0, 1), (2, 3), 8, 7, 4, 5, (9, 10), (0, 1), adjoint=self.adjoint
        )
        gates = (*parity, *termination) if self.adjoint else (*termination, *parity)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    v = torch.arange(16, dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(2048, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for raw, amp in enumerate(amps):
        u, value = raw >> 2, raw & 3
        # Initial marker, terminal, parity and branch are zero; active is one.
        state[((raw << 5) | 8) << 2] = amp
        marker = int(value == 0)
        active = 1 ^ marker
        terminal = marker
        parity = int(u % 2 == 0) & active
        marker ^= int(value % 2 == 0) & (parity ^ 1) & active
        branch = parity ^ marker
        flags = (marker << 4) | (active << 3) | (terminal << 2) | (parity << 1) | branch
        expected[((raw << 5) | flags) << 2] = amp
    circuit = DynamicCircuit(11, inputs=state, dtype=torch.complex128)
    circuit.append(InverseFlags())
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(InverseFlags(adjoint=True))
    restored = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
    print(
        "Complex amplitudes and logical flags correct; two shared quantum work bits zero."
    )
    print(
        "Original reverse layers restored all logical data; this is not a full modular inverse."
    )
    print("Measured forward primitives:", dict(counts))
    print("Independent source estimates: 20 T, five measurements, eleven peak qubits.")


if __name__ == "__main__":
    main()
