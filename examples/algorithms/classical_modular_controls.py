"""Run python -m examples.algorithms.classical_modular_controls.

Source value models and exact append/inverse trajectories without upstream packages.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular_constant_addition import (
    controlled_modular_add_constant_adjoint_ir,
    controlled_modular_add_constant_ir,
    modular_constant_requirements,
)
from flagquantum.compiler.registers.modular_negation import (
    modular_negate_adjoint_ir,
    modular_negate_ir,
    modular_negation_requirements,
)
from flagquantum.compiler.registers.modular_scaled_addition import (
    controlled_modular_scaled_add_adjoint_ir,
    controlled_modular_scaled_add_ir,
    modular_scaled_requirements,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_modular_constant_addition import (
    controlled_modular_add_constant_classical_values,
)
from flagquantum.simulation.classical_modular_negation import (
    modular_negate_classical_value,
)
from flagquantum.simulation.classical_modular_scaled_addition import (
    controlled_modular_scaled_add_classical_values,
)


@dataclass(frozen=True)
class ModularOperation:
    n_qubits: int
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(self.n_qubits, self.instructions))


def _check(
    name: str,
    n_qubits: int,
    forward: tuple[Instruction, ...],
    inverse: tuple[Instruction, ...],
    mapping: list[tuple[int, int]],
    model_t: int,
    model_measurements: int,
) -> None:
    weights = torch.arange(1, len(mapping) + 1, dtype=torch.float64)
    weights = torch.complex(weights, 3 * weights - 5)
    weights /= weights.norm()
    initial = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for (before, after), amplitude in zip(mapping, weights, strict=True):
        initial[before] = amplitude
        expected[after] = amplitude
    circuit = DynamicCircuit(n_qubits, inputs=initial, dtype=torch.complex128)
    circuit.append(ModularOperation(n_qubits, forward))
    computed = run_dynamic(circuit, shots=8, seed=71, strategy="trajectory")
    torch.testing.assert_close(
        computed.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    circuit.append(ModularOperation(n_qubits, inverse))
    restored = run_dynamic(circuit, shots=8, seed=71, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(gate.name for gate in forward)
    actual = counts["t"] + counts["tdg"], counts["measure"]
    assert actual == (model_t, model_measurements)
    print(name, "emitted T/measurements:", actual)
    print(
        name, "pinned model T-equivalent/measurements:", (model_t, model_measurements)
    )


def main() -> None:
    target: tuple[int, ...]
    work_count, slot_count = modular_negation_requirements(3)
    target, work, slots = (
        (0, 1, 2),
        tuple(range(3, 3 + work_count)),
        tuple(range(slot_count)),
    )
    mapping = [
        (x << work_count, int(modular_negate_classical_value(x, 3)) << work_count)
        for x in range(3)
    ]
    _check(
        "ModNeg",
        3 + work_count,
        modular_negate_ir(target, 3, work, slots),
        modular_negate_adjoint_ir(target, 3, work, slots),
        mapping,
        24,
        6,
    )

    work_count, slot_count = modular_constant_requirements(2)
    target, work, slots = (
        (1, 2),
        tuple(range(3, 3 + work_count)),
        tuple(range(slot_count)),
    )
    mapping = []
    for control in (0, 1):
        for x in range(3):
            _, output = controlled_modular_add_constant_classical_values(
                control, x, 1, 3
            )
            mapping.append(
                (
                    ((control << 2) | x) << work_count,
                    ((control << 2) | int(output)) << work_count,
                )
            )
    _check(
        "CModAddK",
        3 + work_count,
        controlled_modular_add_constant_ir(0, target, 1, 3, work, slots),
        controlled_modular_add_constant_adjoint_ir(0, target, 1, 3, work, slots),
        mapping,
        44,
        11,
    )

    work_count, slot_count = modular_scaled_requirements(2)
    left, right, work, slots = (
        (1, 2),
        (3, 4),
        tuple(range(5, 5 + work_count)),
        tuple(range(slot_count)),
    )
    mapping = []
    for control in (0, 1):
        for x in range(4):
            for y in range(3):
                _, _, output = controlled_modular_scaled_add_classical_values(
                    control, x, y, -1, 3
                )
                before = ((control << 4) | (x << 2) | y) << work_count
                after = ((control << 4) | (x << 2) | int(output)) << work_count
                mapping.append((before, after))
    _check(
        "CtrlScaleModAdd",
        5 + work_count,
        controlled_modular_scaled_add_ir(0, left, right, -1, 3, work, slots),
        controlled_modular_scaled_add_adjoint_ir(0, left, right, -1, 3, work, slots),
        mapping,
        96,
        24,
    )
    print("All complex amplitudes restored; each work register ends in zero")


if __name__ == "__main__":
    main()
