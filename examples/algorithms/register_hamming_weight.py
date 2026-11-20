"""Coherently copy a population count and clear source-tree garbage.

Run: python -m examples.algorithms.register_hamming_weight
Expected: every four-bit input amplitude preserved, count copied, six work
wires zero. No reference package is required. Counts describe the full path.
"""

from collections import Counter

import torch

import flagquantum as fq
from flagquantum.compiler.registers.bits.bitwise import xor_ir
from flagquantum.compiler.registers.bits.hamming_weight_component import (
    hamming_weight_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


class PopulationCount:
    """Bind four data bits, three junk, three temporary output and three result bits."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        compute = bind_register_program(
            hamming_weight_program(4),
            ((0, 1, 2, 3),),
            clean=(4, 5, 6, 7, 8, 9),
            allow_workspace_outputs=True,
        )
        erase = bind_register_program(
            hamming_weight_program(4, adjoint=True),
            compute.outputs,
            input_ownership=tuple(
                (wire, False) for word in compute.outputs[1:] for wire in word
            ),
        )
        copy = xor_ir(compute.outputs[2], (10, 11, 12))
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                (*compute.circuit.instructions, *copy, *erase.circuit.instructions),
            )
        )


def main() -> None:
    coeff = torch.arange(16, dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 13, dtype=torch.complex128)
    data = torch.arange(16, dtype=torch.int64)
    state[data << 9] = amps
    counts = torch.tensor([value.bit_count() for value in range(16)])
    expected = torch.zeros_like(state)
    expected[(data << 9) | counts] = amps
    circuit = DynamicCircuit(13, inputs=state, dtype=torch.complex128)
    circuit.append(PopulationCount())
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )
    gates = Counter(g.name for g in circuit.to_ir().instructions)
    print("All input/count complex amplitudes verified; six temporary wires zero.")
    print("Measured primitive gates:", dict(gates))
    print("Independent source T/measurement model: 12 T, 3 measurements.")
    print("Source tree junk: 3; temporary count: 3; retained result: 3.")


if __name__ == "__main__":
    main()
