"""Run python -m examples.algorithms.classical_measurement_values.

Expected: X-measurement numerical outcomes retain their branch phases; copied
work clears coherently with 0 T, 1 measurement, 1 work wire and 3 declared wires.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    MeasurementCall,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_distributions import draw_classical_value
from flagquantum.simulation.classical_measurement import (
    measure_x_basis_phase,
    measure_x_classical_values,
)
from flagquantum.simulation.classical_registers import (
    bind_integer_register_values,
    collect_integer_register_value,
)
from flagquantum.simulation.classical_updates import (
    commit_classical_outputs,
    commit_phased_classical_outputs,
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


def evaluate_measurement(
    value: int, rng: np.random.Generator
) -> tuple[object, complex]:
    """Resolve and bind the output before using its deferred measured phase."""
    outputs = measure_x_classical_values(value)
    contribution = measure_x_basis_phase(value)
    outcomes: dict[tuple[str, tuple[int, ...]], object] = {}

    def assign(values: Mapping[str, object]) -> None:
        bound = bind_integer_register_values(
            "c",
            values,
            1,
            resolve_distribution=lambda distribution: draw_classical_value(
                distribution, instance_id=0, rng=rng
            ),
        )
        outcomes[("c", ())] = collect_integer_register_value(bound, 1)

    phase = commit_phased_classical_outputs(
        outputs, contribution, assign, phase=1.0, outcomes=outcomes
    )
    return outcomes[("c", ())], phase


def main() -> None:
    rng = np.random.default_rng(47)
    for value in (0, 1):
        samples = []
        for _ in range(8):
            outcome, phase = evaluate_measurement(value, rng)
            assert phase == (-1 if value == 1 and outcome == 1 else 1)
            samples.append((outcome, phase))
        print("Basis value, sampled outcomes and phases:", value, samples)
    pure_writes: dict[tuple[int, ...], object] = {}

    def assign_pure(values: Mapping[str, object]) -> None:
        pure_writes.update(bind_integer_register_values("q", values, 1))

    commit_classical_outputs({"q": 0}, None, assign_pure, operation_name="X")
    try:
        commit_classical_outputs({"q": 1}, -1, assign_pure, operation_name="Z")
    except ValueError:
        assert pure_writes == {(): 1}
    else:
        raise AssertionError("Pure classical mode must reject a phase after assignment")
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
