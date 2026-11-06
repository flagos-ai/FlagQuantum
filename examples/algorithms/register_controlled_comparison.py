"""Use circuit.append(op) for signed/unsigned controlled greater-than.

Run: python -m examples.algorithms.register_controlled_comparison
Expected: all complex input/control/target amplitudes correct, clean seven-wire
workspace and double-XOR restoration for both control polarities. No references.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.controlled_comparison import (
    controlled_greater_ir,
    controlled_greater_requirements,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledGreater:
    """Bind two-bit operands and explicit sum/sign/control workspace."""

    signed: bool
    control_value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        q, c = controlled_greater_requirements(2)
        gates = controlled_greater_ir(
            0,
            (1, 2),
            (3, 4),
            5,
            tuple(range(6, 6 + q)),
            tuple(range(c)),
            signed=self.signed,
            control_value=self.control_value,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    q, c = controlled_greater_requirements(2)
    data = torch.arange(64, dtype=torch.int64)
    control = data >> 5
    left = (data >> 3) & 3
    right = (data >> 1) & 3
    target = data & 1
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    n_qubits = 6 + q
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[data << q] = amps
    for signed in (False, True):
        a = left - ((left >> 1) << 2) if signed else left
        b = right - ((right >> 1) << 2) if signed else right
        for cv in (0, 1):
            mapped = (data & ~1) | (target ^ ((control == cv) & (a > b)))
            expected = torch.zeros_like(state)
            expected[mapped << q] = amps
            operation = ControlledGreater(signed, cv)
            circuit = DynamicCircuit(
                n_qubits, inputs=state, dtype=torch.complex128
            ).append(operation)
            result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(32, -1),
                expected.expand(32, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            restored = run_dynamic(
                circuit.append(operation), shots=8, seed=41, strategy="trajectory"
            )
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                state.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                f"signed={signed}, control={cv}: all complex amplitudes correct, work zero, double XOR verified"
            )
            print(
                "Quantum work:",
                q,
                "Measurement slots:",
                c,
                "Measured gates:",
                dict(counts),
            )


if __name__ == "__main__":
    main()
