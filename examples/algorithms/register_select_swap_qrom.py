"""XOR a QROM entry through circuit.append(op) using a borrowed bank.

Run: python -m examples.algorithms.register_select_swap_qrom
"""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.select_swap_qrom import select_swap_qrom_ir
from flagquantum.compiler.registers.source_qroam_clean import QroamCleanTable
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class SelectSwapLookup:
    """Bind a controlled one-bit lookup and a two-wire borrowed bank."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        table = QroamCleanTable((4,), (1,), ((0, 1, 1, 0),))
        gates = select_swap_qrom_ir(
            table, (0,), ((1, 2),), ((3,),), (((4,), (5,)),), (6,), (0,), dirty=True
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    initial = torch.zeros(128, dtype=torch.complex128)
    # Distinct borrowed states are correlated with the control and address.
    initial[82] = 0.6j  # Enabled, address 1, target 0, bank 01.
    initial[44] = 0.8  # Disabled, address 2, target 1, bank 10.
    expected = torch.zeros_like(initial)
    expected[90], expected[44] = initial[82], initial[44]
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    op = SelectSwapLookup()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), atol=1e-12, rtol=0
    )
    print("Lookup XOR applied; borrowed bank restored with its coherence.")


if __name__ == "__main__":
    main()
