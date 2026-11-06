"""Apply fixed-point phase kickback through circuit.append(op).

Run: python -m examples.algorithms.register_phase_addition
"""

from cmath import exp
from dataclasses import dataclass
from math import tau

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase_addition import phase_add_ir
from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledPhaseKickback:
    """Subtract a shifted two-bit fraction on the control-zero branch."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gradient = (3, 4, 5)
        gates = (
            *phase_gradient_state_ir(gradient),
            *phase_add_ir(
                (1, 2),
                gradient,
                (6, 7, 8, 9),
                (0, 1, 2, 3),
                right_shift=1,
                sign=-1,
                control=0,
                control_value=0,
            ),
            *phase_gradient_state_ir(gradient, inverse=True),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(1024, dtype=torch.complex128)
    initial[3 << 7] = 0.6j
    initial[6 << 7] = 0.8
    expected = initial.clone()
    expected[3 << 7] *= exp(1j * tau * 3 / 8)
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128)
    op = ControlledPhaseKickback()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Controlled kickback phase verified; gradient and work returned to zero.")


if __name__ == "__main__":
    main()
