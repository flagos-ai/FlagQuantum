"""Run python -m examples.algorithms.classical_contiguous_index_values."""

from dataclasses import dataclass
from itertools import product

import torch

import flagquantum as fq
from flagquantum.compiler.registers.contiguous_index import (
    contiguous_index_columns_ir,
    source_contiguous_index_cost,
)
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_contiguous_index import (
    contiguous_index_classical_values,
)


@dataclass(frozen=True)
class ContiguousIndex:
    """XOR a triangular index into an occupied target using one clean work bit."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                contiguous_index_columns_ir((0,), (1,), (2, 3), (4,)),
            )
        )


def main() -> None:
    grid = torch.arange(1, 33, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(16), (0, 1)):
        mu, nu, target = data >> 3, (data >> 2) & 1, data & 3
        index = contiguous_index_classical_values(mu, nu)[2]
        updated = (data & ~3) | (target ^ (index & 3))
        initial[(data << 2) | spectator] = weights[2 * data + spectator]
        expected[(updated << 2) | spectator] = weights[2 * data + spectator]
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    op = ContiguousIndex()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=43, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    report = resource_model_report(
        source_contiguous_index_cost(1, 2), emitted_ir=circuit.to_ir()
    )
    assert report["t_equivalents"] == report["emitted_ir"]["t_count"] == 4
    assert report["emitted_ir"]["unconditional"]["measure"] == 1
    circuit.append(op)
    restored = run_dynamic(circuit, shots=8, seed=47, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print("All complex amplitudes agree; work bit zero; repeated XOR restores input")
    print("Emitted: 4 T, 1 measurement, 1 clean work bit, 6 declared wires")
    print("Source model: 1 Toffoli, 4 T equivalents, 0 measurements, 4 logical wires")
    print(
        "The source root has no gate decomposition; these are separate evidence bases"
    )


if __name__ == "__main__":
    main()
