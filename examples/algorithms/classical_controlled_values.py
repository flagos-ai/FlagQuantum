"""Run python -m examples.algorithms.classical_controlled_values."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_controlled_values import (
    controlled_basis_phase,
    controlled_classical_values,
)
from flagquantum.simulation.classical_pauli_values import (
    diagonal_classical_values,
    x_classical_value,
    z_basis_phase,
)


def x_child(**values: object) -> dict[str, object]:
    target = values["q"]
    assert isinstance(target, (int, np.integer))
    return {"q": x_classical_value(target)}


def z_child(**values: object) -> dict[str, object]:
    return diagonal_classical_values(**values)


def z_phase(**values: object) -> int:
    target = values["q"]
    assert isinstance(target, (int, np.integer))
    return z_basis_phase(target)


@dataclass(frozen=True)
class WordControlledPaulis:
    """Apply X then Z for word two, or their exact reverse order."""

    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        kinds: tuple[Literal["x", "z"], ...] = (
            ("z", "x") if self.inverse else ("x", "z")
        )
        gates = tuple(
            gate
            for kind in kinds
            for gate in pauli_gate_ir(
                kind,
                2,
                controls=(0, 1),
                values=(1, 0),
                workspace=(3,),
                classical_bits=(0,),
            )
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    patterns = (np.array(2),)
    grid = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        values = controlled_classical_values(
            {"ctrl": word, "q": target}, ("ctrl",), ("q",), patterns, x_child
        )
        assert isinstance(values, dict)
        phase = controlled_basis_phase(values, ("ctrl",), ("q",), patterns, z_phase)
        output = controlled_classical_values(
            values, ("ctrl",), ("q",), patterns, z_child
        )
        assert isinstance(output, dict) and isinstance(output["q"], int)
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | (output["q"] << 2) | spectator
        initial[before] = weights[(word << 2) | (target << 1) | spectator]
        # This child is Z: inactive None means identity; active phases are +/-1.
        expected[after] = (1 if phase is None else phase) * initial[before]
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(WordControlledPaulis())
    result = run_dynamic(circuit, shots=8, seed=79, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 8 and counts["measure"] == 2
    circuit.append(WordControlledPaulis(inverse=True))
    restored = run_dynamic(circuit, shots=8, seed=83, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print(
        "Controlled values and phases agree on complete complex amplitudes and inverse"
    )
    print("Work zero; original controls and spectator preserved")
    print("Emitted: 8 T, 2 measurements, 1 reused work, 5 declared wires")
    print("Source model: 8 T equivalents, 2 measurements, 4 peak logical wires")


if __name__ == "__main__":
    main()
