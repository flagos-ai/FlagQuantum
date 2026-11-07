"""Run python -m examples.algorithms.classical_controlled_bits."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.bitwise import fanout_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.classical_controlled_bits import (
    cnot_classical_values,
    fanout_classical_values,
    toffoli_classical_values,
)


@dataclass(frozen=True)
class ControlledBits:
    """Compose source primitive CNOT, atomic Toffoli and a two-target fanout."""

    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = (
            Instruction("cx", (0, 2)),
            Instruction("ccx", (0, 1, 2)),
            *fanout_ir(0, (1, 2)),
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, tuple(reversed(gates)) if self.inverse else gates
            )
        )


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(8), (0, 1)):
        a, b, target = data >> 2, (data >> 1) & 1, data & 1
        target = int(cnot_classical_values(a, target)[1])
        target = int(toffoli_classical_values(np.array([a, b]), target)[1])
        targets = fanout_classical_values(a, (b << 1) | int(target), 2)[1]
        updated = (a << 2) | int(targets)
        expected[(updated << 1) | spectator] = initial[(data << 1) | spectator]
    circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(ControlledBits())
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert Counter(g.name for g in circuit.to_ir().instructions) == {"cx": 4, "ccx": 1}
    circuit.append(ControlledBits(inverse=True))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print("Complete complex action and inverse pass; spectator preserved; no workspace")
    print("Emitted: 4 CX, 1 atomic CCX, 0 explicit T, 0 measurements")
    print(
        "Source modern model: 2 logical Clifford components, 1 Toffoli, 4 T equivalents"
    )
    print("The atomic CCX has not been synthesized into a Clifford+T gate sequence")


if __name__ == "__main__":
    main()
