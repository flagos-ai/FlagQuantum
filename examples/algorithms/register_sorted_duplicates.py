"""Flag sorted-list duplicates coherently through circuit.append(op).

Run: python -m examples.algorithms.register_sorted_duplicates
"""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.sorted_duplicates import sorted_duplicates_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class SortedDuplicates:
    """Bind two one-bit words and a flag, with five clean workspace wires."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates, _ = sorted_duplicates_ir(((0,), (1,)), 2, (3, 4, 5, 6, 7), (0,))
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[0] = 0.6j
    initial[64] = 0.8
    expected = initial.clone()
    expected[32], expected[0] = initial[0], 0
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    op = SortedDuplicates()
    circuit.append(op)
    result = run_dynamic(circuit, shots=4, seed=12, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(4, -1), atol=1e-12, rtol=0
    )
    print("Flag toggled for [0, 0]; [0, 1] preserved; workspace returned to zero.")


if __name__ == "__main__":
    main()
