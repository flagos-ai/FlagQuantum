"""Phase a word by its Hamming weight through circuit.append(op).

Run: python -m examples.algorithms.register_hamming_weight_phasing
"""

from cmath import exp
from dataclasses import dataclass
from math import pi

import torch

import flagquantum as fq
from flagquantum.compiler.registers.hamming_weight_phasing import (
    hamming_weight_phasing_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class HammingPhase:
    """Bind three source wires and three clean count-tree wires."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = hamming_weight_phasing_ir(
            (0, 1, 2), (3,), (4, 5), (0,), exponent=-0.375
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[0], initial[3 << 3] = 0.8, 0.6j
    expected = initial.clone()
    expected[3 << 3] *= exp(-0.75j * pi)
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    op = HammingPhase()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Hamming-weight phase verified; count and junk returned to zero.")


if __name__ == "__main__":
    main()
