"""Run python -m examples.algorithms.classical_and_values."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.control.multi_control import multi_and_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_and import multi_and_classical_values


@dataclass(frozen=True)
class Predicate:
    """Compute logical prefix outputs, or erase them with two measurements."""

    uncompute: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = multi_and_ir(
            (0, 1, 2),
            (0, 1, 0),
            (3, 4),
            classical_bits=(0, 1) if self.uncompute else (),
            uncompute=self.uncompute,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(8), (0, 1)):
        controls = np.array([(data >> i) & 1 for i in (2, 1, 0)], dtype=np.uint8)
        output = multi_and_classical_values(controls, (0, 1, 0))
        junk = output["junk"]
        assert isinstance(junk, np.ndarray) and junk.shape == (1,)
        prefix = (int(junk[0]) << 1) | int(output["target"])
        initial[(data << 3) | spectator] = weights[2 * data + spectator]
        expected[(data << 3) | (prefix << 1) | spectator] = weights[
            2 * data + spectator
        ]
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(Predicate())
    result = run_dynamic(circuit, shots=1, seed=53, strategy="trajectory")
    torch.testing.assert_close(result.final_states[0], expected, rtol=0, atol=2e-12)
    circuit.append(Predicate(uncompute=True))
    restored = run_dynamic(circuit, shots=8, seed=59, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 8 and counts["measure"] == 2
    print("All complex amplitudes and spectator preserved; both prefix outputs erased")
    print("Emitted compute + erase: 8 T, 2 measurements, 2 logical outputs, 6 wires")
    print(
        "Source model: compute 8 T equivalents/0 measurements; erase 0 T/2 measurements"
    )


if __name__ == "__main__":
    main()
