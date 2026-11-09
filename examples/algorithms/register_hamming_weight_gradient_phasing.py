"""Phase by Hamming weight with a reusable gradient via circuit.append(op).

Run: python -m examples.algorithms.register_hamming_weight_gradient_phasing
"""

from cmath import exp
from dataclasses import dataclass
from math import pi

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase.hamming_weight_phasing import (
    hamming_weight_gradient_phasing_ir,
)
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class HammingGradientPhase:
    """Phase a two-bit word at exponent 1/8 with a three-bit gradient."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gradient = (5, 6, 7)
        gates = (
            *phase_gradient_state_ir(gradient, exponent=-1),
            *hamming_weight_gradient_phasing_ir(
                (0, 1),
                (2,),
                (3, 4),
                gradient,
                (8, 9),
                (0,),
                (0, 1),
                exponent=0.125,
                epsilon=0.5,
            ),
            *phase_gradient_state_ir(gradient, exponent=-1, inverse=True),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(1024, dtype=torch.complex128)
    initial[0], initial[3 << 8] = 0.8, 0.6j
    expected = initial.clone()
    expected[3 << 8] *= exp(1j * pi / 4)
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128)
    op = HammingGradientPhase()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Hamming-weight kickback verified; gradient and work returned to zero.")


if __name__ == "__main__":
    main()
