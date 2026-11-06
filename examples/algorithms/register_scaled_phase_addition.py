"""Apply a scaled phase and clear the gradient through circuit.append(op).

Run: python -m examples.algorithms.register_scaled_phase_addition
"""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.scaled_phase_addition import scaled_phase_add_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ScaledPhaseKickback:
    """Bind a three-bit fraction and a four-bit gradient at coefficient 0.75."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gradient = (3, 4, 5, 6)
        gates = (
            *phase_gradient_state_ir(gradient),
            *scaled_phase_add_ir(
                (0, 1, 2),
                gradient,
                (7, 8, 9),
                (0, 1, 2),
                input_fraction_bits=3,
                coefficient=0.75,
                coefficient_width=3,
                coefficient_fraction_bits=3,
            ),
            *phase_gradient_state_ir(gradient, inverse=True),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(1024, dtype=torch.complex128)
    initial[0] = 0.8
    initial[3 << 7] = 0.6j
    expected = initial.clone()
    # The two source terms contribute 3 and 1 on this input.
    expected[3 << 7] *= -1j
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128)
    op = ScaledPhaseKickback()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Scaled phase verified; gradient and work returned to zero.")


if __name__ == "__main__":
    main()
