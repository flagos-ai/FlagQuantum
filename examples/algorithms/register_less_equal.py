"""Run unequal-width tree comparison with circuit.append(op).

Run: python -m examples.algorithms.register_less_equal
Expected: all complex input and target amplitudes match unsigned x<=y,
all seven work wires return to zero, and a second comparison restores inputs.
No reference packages are required.
"""

from collections import Counter

import torch

import flagquantum as fq
from flagquantum.compiler.registers.comparison.comparison_resource_model import (
    source_less_equal_cost,
)
from flagquantum.compiler.registers.comparison.less_equal import (
    less_equal_ir,
    less_equal_requirements,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


class LessEqual:
    """Bind two-bit x and three-bit y, one XOR target and explicit work."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        quantum, classical = less_equal_requirements(2, 3)
        gates = less_equal_ir(
            (0, 1), (2, 3, 4), 5, tuple(range(6, 6 + quantum)), tuple(range(classical))
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    quantum, classical = less_equal_requirements(2, 3)
    n_qubits = 6 + quantum
    data = torch.arange(64, dtype=torch.int64)
    values = data >> 1
    x = values >> 3
    y = values & 7
    target = data & 1
    mapped = (values << 1) | (target ^ (x <= y))
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[data << quantum] = amps
    expected = torch.zeros_like(state)
    expected[mapped << quantum] = amps
    operation = LessEqual()
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128)
    circuit.append(operation)
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )
    report = estimate_ftqc_resources(
        source_less_equal_cost(2, 3), emitted_ir=circuit.to_ir()
    )
    assert report.emitted is not None
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(operation)
    restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=1e-12
    )
    print(
        "Two-bit x <= three-bit y: every complex amplitude correct; clean work and double XOR verified"
    )
    print("Clean quantum work:", quantum, "Classical measurement slots:", classical)
    print("Measured gates for one comparison:", dict(counts))
    print("Source model categories:", report.logical.gates)
    print("Model peak logical qubits:", report.logical.peak_logical_qubits)
    print("Model T equivalents:", report.logical.gates.total_t_count())
    print("Emitted T/Tdag:", report.emitted.t_count)


if __name__ == "__main__":
    main()
