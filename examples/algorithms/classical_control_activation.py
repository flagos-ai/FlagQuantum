"""Run python -m examples.algorithms.classical_control_activation."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.pauli_gate import pauli_gate_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_control_activation import (
    classical_controls_active,
)


@dataclass(frozen=True)
class WordControlledX:
    """Flip a target only for the two-bit word two, using shared predicate work."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = pauli_gate_ir(
            "x", 2, controls=(0, 1), values=(1, 0), workspace=(3,), classical_bits=(0,)
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        active = classical_controls_active(
            (word,), (np.array(2),), symbolic_parameters=(2,)
        )
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | ((target ^ active) << 2) | spectator
        initial[before] = weights[(word << 2) | (target << 1) | spectator]
        expected[after] = initial[before]
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    op = WordControlledX()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=71, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 4 and counts["measure"] == 1
    circuit.append(op)
    restored = run_dynamic(circuit, shots=8, seed=73, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print("Word activation matches complete complex controlled action and inverse")
    print("Predicate work zero; all input words, targets and spectator preserved")
    print("Emitted: 4 T, 1 measurement, 1 work, 5 declared wires")
    print("Source model: 4 T equivalents, 1 measurement, 4 peak logical wires")


if __name__ == "__main__":
    main()
