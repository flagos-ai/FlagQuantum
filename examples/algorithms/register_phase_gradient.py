"""Apply and reverse a controlled phase gradient through circuit.append(op).

Run: python -m examples.algorithms.register_phase_gradient
"""

from cmath import exp
from dataclasses import dataclass
from math import pi

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase.phase_gradient import (
    phase_gradient_unitary_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledPhaseGradient:
    """Bind a three-bit phase gradient, one control and one clean work wire."""

    exponent: float = -0.75

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = phase_gradient_unitary_ir(
            (0, 1, 2), exponent=self.exponent, control=3, work=4, classical_bit=0
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[6] = 0.6j  # Gradient 1, control 1, work 0.
    initial[20] = 0.8  # Gradient 5, control 0, work 0.
    expected = initial.clone()
    expected[6] *= exp(-0.75j * pi / 4)
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    op = ControlledPhaseGradient()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    circuit.append(ControlledPhaseGradient(-op.exponent))
    restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Controlled phase verified; the inverse restored the input and clean work.")


if __name__ == "__main__":
    main()
