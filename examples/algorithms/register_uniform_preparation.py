"""Append controlled uniform preparation and compare source model with lowered IR.

Run python -m examples.algorithms.register_uniform_preparation.
"""

from collections import Counter
from math import acos, sin, sqrt

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.basis_translation import translate
from flagquantum.compiler.registers.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    values = (1, 0)
    target, work, slots = (2, 3, 4), (5, 6, 7, 8), (0, 1, 2)
    gates = uniform_preparation_ir(
        6, target, work, slots, controls=(0, 1), control_values=values
    )
    lowered: list[Instruction] = []
    for gate in gates:
        if gate.name == "ch":
            translated = translate(
                gate, can_run=lambda g: g.name in {"h", "s", "sdg", "t", "tdg", "cx"}
            )
            assert translated is not None
            lowered.extend(translated)
        else:
            lowered.append(gate)
    initial = torch.zeros(512, dtype=torch.complex128)
    amplitudes = torch.tensor([1, 1j, -1, 1], dtype=torch.complex128) / 2
    initial[::128] = amplitudes
    desired = initial.clone()
    probability = 3 / 4
    theta = acos(1 / 3)
    phase = sqrt(probability) * complex(1 / (2 * probability), sin(theta))
    desired[256:352:16] = amplitudes[2] * phase / sqrt(6)
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(9, tuple(lowered)))
    result = run_dynamic(circuit, shots=8, seed=97, strategy="batched")
    torch.testing.assert_close(
        result.final_states, desired.expand_as(result.final_states), atol=4e-12, rtol=0
    )
    model = source_uniform_preparation_cost(6, control_values=values)
    report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    reverse = uniform_preparation_ir(
        6, target, work, slots, controls=(0, 1), control_values=values, adjoint=True
    )
    circuit.append(FieldOperation(9, reverse))
    restored = run_dynamic(circuit, shots=8, seed=101, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=4e-12,
        rtol=0,
    )
    primitives = Counter(g.name for g in lowered)
    print("Source model categories", report.logical.gates)
    print(
        "Model T equivalents excluding rotations",
        report.logical.gates.total_t_count(ts_per_rotation=0),
    )
    print("Model rotations", report.logical.gates.rotation)
    print("Emitted T/Tdag", report.emitted.t_count)
    print("Remaining unsynthesized Rz", primitives["rz"])
    print("Measurements", primitives["measure"])
    print("Peak logical qubits", report.logical.peak_logical_qubits)
    print(
        "Absolute complex phase, coherent inactive controls, clean work and inverse verified"
    )


if __name__ == "__main__":
    main()
