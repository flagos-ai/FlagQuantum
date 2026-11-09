"""Apply direct variable Z phases through circuit.append(op).

Run: python -m examples.algorithms.register_qvr_zpow
"""

from cmath import exp
from dataclasses import dataclass
from math import tau

import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase.qvr_zpow import qvr_zpow_ir
from flagquantum.core.ir import CircuitIR


@dataclass(frozen=True)
class VariableZPhase:
    """Rotate a three-bit fraction with coefficient 0.75."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = qvr_zpow_ir(
            (0, 1, 2), cost_fraction_bits=3, coefficient=0.75, epsilon=0.01
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[0], initial[3] = 0.8, 0.6j
    expected = initial.clone()
    expected[3] *= exp(1j * tau * 0.75 * 3 / 8)
    circuit = fq.Circuit(3, inputs=initial, dtype=torch.complex128)
    op = VariableZPhase()
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    print("Direct QVR preserved the complete complex phase with no work register.")


if __name__ == "__main__":
    main()
