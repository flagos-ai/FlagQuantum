"""Select a QVR gradient size and apply kickback through circuit.append(op).

Run: python -m examples.algorithms.register_qvr_phase_gradient
"""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    qvr_phase_format,
    qvr_phase_gradient_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class VariablePhase:
    """Bind a three-bit cost word and the gradient selected at epsilon 0.25."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        fmt = qvr_phase_format(3, 3, 0.75, 0.25)
        gradient = tuple(range(3, 3 + fmt.gradient_width))
        gates = (
            *phase_gradient_state_ir(gradient, exponent=-1),
            *qvr_phase_gradient_ir(
                (0, 1, 2),
                gradient,
                (7, 8, 9),
                (0, 1, 2),
                cost_fraction_bits=3,
                coefficient=0.75,
                epsilon=0.25,
            ),
            *phase_gradient_state_ir(gradient, exponent=-1, inverse=True),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(1024, dtype=torch.complex128)
    initial[0] = 0.8
    initial[3 << 7] = 0.6j
    expected = initial.clone()
    expected[3 << 7] *= 1j  # The two retained increments sum to four modulo 16.
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128)
    op = VariablePhase()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("QVR selected four gradient bits and restored the gradient and work.")


if __name__ == "__main__":
    main()
