"""Apply Z and Rz kickback with a reusable gradient through circuit.append(op).

Run: python -m examples.algorithms.register_phase_gradient_rotations
"""

from cmath import exp
from dataclasses import dataclass
from math import tau

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.rz_phase_gradient import rz_via_phase_gradient_ir
from flagquantum.compiler.registers.zpow_phase_gradient import (
    zpow_via_phase_gradient_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class GradientRotations:
    """Apply quantized Z then variable Rz with the same three-bit gradient."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gradient = (3, 4, 5)
        gates = (
            *phase_gradient_state_ir(gradient, exponent=-1),
            *zpow_via_phase_gradient_ir(
                0, gradient, (6, 7, 8), (9, 10), (0, 1), exponent=0.75
            ),
            *rz_via_phase_gradient_ir(0, (1, 2), gradient, (9, 10), (0, 1)),
            *phase_gradient_state_ir(gradient, exponent=-1, inverse=True),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(2048, dtype=torch.complex128)
    initial[1 << 8] = 0.6j  # q=0, angle=1
    initial[6 << 8] = 0.8  # q=1, angle=2
    expected = initial.clone()
    expected[1 << 8] *= exp(-1j * tau / 8)
    expected[6 << 8] *= exp(1j * tau * (3 + 2) / 8)
    circuit = DynamicCircuit(11, inputs=initial, dtype=torch.complex128)
    op = GradientRotations()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Z and Rz phases verified; the shared gradient and work returned to zero.")


if __name__ == "__main__":
    main()
