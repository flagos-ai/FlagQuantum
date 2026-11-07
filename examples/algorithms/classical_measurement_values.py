"""Run python -m examples.algorithms.classical_measurement_values.

Expected: X-measurement numerical outcomes retain their branch phases; copied
work clears coherently with 0 T, 1 measurement, 1 work wire and 3 declared wires.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    MeasurementCall,
)
from flagquantum.compiler.registers.measurement import measure_x_clear_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_distributions import draw_classical_value
from flagquantum.simulation.classical_measurement import (
    apply_classical_phase,
    measure_x_basis_phase,
    measure_x_classical_values,
)
from flagquantum.simulation.classical_registers import (
    bind_integer_register_values,
    collect_integer_register_value,
)


@dataclass(frozen=True)
class ClearCopiedWork:
    """Copy data to clean work, then erase it with measured phase feedback."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = (
            Instruction("cx", (0, 1)),
            *measure_x_clear_ir(1, 0),
            Instruction("z", (0,), metadata={"conditions": ((0, 1),)}),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    rng = np.random.default_rng(47)
    for value in (0, 1):
        samples = []
        for _ in range(8):
            assignments = bind_integer_register_values(
                "c",
                measure_x_classical_values(value),
                1,
                resolve_distribution=lambda distribution: draw_classical_value(
                    distribution, instance_id=0, rng=rng
                ),
            )
            outcome = collect_integer_register_value(assignments, 1)
            phase = apply_classical_phase(
                1.0, measure_x_basis_phase(value), {("c", ()): outcome}
            )
            assert phase == (-1 if value == 1 and outcome == 1 else 1)
            samples.append((outcome, phase))
        print("Basis value, sampled outcomes and phases:", value, samples)
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[[0, 1, 4, 5]] = torch.tensor(
        [1 + 2j, 3 - 4j, -2 + 1j, 4 + 3j], dtype=torch.complex128
    )
    initial /= initial.norm()
    circuit = DynamicCircuit(3, inputs=initial, dtype=torch.complex128)
    circuit.append(ClearCopiedWork())
    result = run_dynamic(circuit, shots=8, seed=43, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["measure"] == 1 and counts["t"] + counts["tdg"] == 0
    print("MeasureX leaf model:", query_qec_gate_cost(MeasurementCall("x")))
    print("Emitted copy/erase sequence:", dict(counts))
    print("Full complex amplitudes preserved; 1 work wire zero; 3 declared wires")


if __name__ == "__main__":
    main()
