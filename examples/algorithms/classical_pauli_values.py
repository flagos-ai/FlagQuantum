"""Run python -m examples.algorithms.classical_pauli_values."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import torch

import flagquantum as fq
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.classical_pauli_values import (
    cz_basis_phase,
    diagonal_classical_values,
    x_classical_value,
    z_basis_phase,
)


@dataclass(frozen=True)
class PauliSequence:
    """Compose X, Z and CZ through the existing source-controlled Pauli emitter."""

    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = (
            *pauli_gate_ir("x", 0),
            *pauli_gate_ir("z", 1),
            *pauli_gate_ir("z", 1, controls=(0,), values=(1,)),
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, tuple(reversed(gates)) if self.inverse else gates
            )
        )


def main() -> None:
    grid = torch.arange(1, 9, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for left, right, spectator in product((0, 1), repeat=3):
        updated = int(x_classical_value(left))
        values = diagonal_classical_values(q1=updated, q2=right)
        phase = z_basis_phase(right) * cz_basis_phase(values["q1"], values["q2"])
        before = (left << 2) | (right << 1) | spectator
        after = (updated << 2) | (right << 1) | spectator
        expected[after] = phase * initial[before]
    circuit = fq.Circuit(3, inputs=initial, dtype=torch.complex128)
    circuit.append(PauliSequence())
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert Counter(g.name for g in circuit.to_ir().instructions) == {
        "x": 1,
        "z": 1,
        "cz": 1,
    }
    circuit.append(PauliSequence(inverse=True))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print("Values and basis phases reproduce full complex amplitudes and exact inverse")
    print("Emitted: 1 X, 1 Z, 1 CZ, 0 T, 0 measurements, 0 work")
    print("Source model: 3 Clifford components, 0 T equivalents; spectator preserved")


if __name__ == "__main__":
    main()
