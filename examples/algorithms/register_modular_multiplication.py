"""Coherent controlled modular multiplication with exact source CSwap cleanup.

Run: python -m examples.algorithms.register_modular_multiplication
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular_multiplication import (
    controlled_modular_multiply_constant_adjoint_ir,
    controlled_modular_multiply_constant_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularMultiplication:
    """Bind a coherent control, two-bit residue modulo three, and clean work."""

    value: int = 2
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        emit = (
            controlled_modular_multiply_constant_adjoint_ir
            if self.inverse
            else controlled_modular_multiply_constant_ir
        )
        gates = emit(0, (1, 2), self.value, 3, tuple(range(3, 16)), (0, 1, 2, 3))
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    labels = [(c, x) for c in (0, 1) for x in range(3)]
    coeff = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 16, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (c, x), amp in zip(labels, amps, strict=True):
        y = (2 * x) % 3 if c else x
        state[((c << 2) | x) << 13] = amp
        expected[((c << 2) | y) << 13] = amp
    for inverse in (False, True):
        circuit = DynamicCircuit(16, inputs=state, dtype=torch.complex128)
        circuit.append(ModularMultiplication(inverse=inverse))
        result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(8, -1),
            expected.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        circuit.append(ModularMultiplication(inverse=not inverse))
        restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            restored.final_states.reshape(8, -1),
            state.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            "inverse",
            inverse,
            ": coherent complex amplitudes correct; thirteen work wires zero; original exact inverse restored input",
        )
        print("Measured primitive gates:", dict(counts))
    print(
        "Source exact 7T-CSwap convention: 206 T; default 4T-CSwap model: 200 T; measurements: 48."
    )
    print(
        "Work: thirteen quantum wires; four reused classical slots; total sixteen qubits."
    )


if __name__ == "__main__":
    main()
