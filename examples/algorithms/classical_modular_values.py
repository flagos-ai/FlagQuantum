"""Run python -m examples.algorithms.classical_modular_values.

Keep source classical diagnostics separate from clean coherent modular execution.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.modular_addition import modular_add_ir
from flagquantum.compiler.registers.modular.modular_subtraction import (
    modular_subtract_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_modular_addition import (
    modular_add_classical_values,
)
from flagquantum.simulation.classical_modular_subtraction import (
    modular_subtract_classical_values,
)


@dataclass(frozen=True)
class ModularOperation:
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(10, self.instructions))


def main() -> None:
    left, right, work, slots = (0, 1), (2, 3), tuple(range(4, 10)), (0, 1)
    add = modular_add_ir(left, right, 2, work, slots)
    subtract = modular_subtract_ir(left, right, 2, work, slots)
    initial = torch.zeros(1024, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    amplitudes = torch.tensor([0.8, 0.36j, 0.48j], dtype=torch.complex128)
    for (a, b), amplitude in zip(((0, 0), (1, 0), (1, 1)), amplitudes, strict=True):
        initial[((a << 2) | b) << 6] = amplitude
        preserved, output = modular_add_classical_values(a, b, 2)
        assert isinstance(output, int)
        assert modular_subtract_classical_values(preserved, output, 2) == (a, b)
        expected[((a << 2) | output) << 6] = amplitude
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128)
    circuit.append(ModularOperation(add))
    forward = run_dynamic(circuit, shots=8, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    circuit.append(ModularOperation(subtract))
    reverse = run_dynamic(circuit, shots=8, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    for name, gates, model_t, model_measurements in (
        ("ModAdd", add, 28, 7),
        ("ModSub", subtract, 36, 9),
    ):
        counts = Counter(gate.name for gate in gates)
        actual = counts["t"] + counts["tdg"], counts["measure"]
        assert actual == (model_t, model_measurements)
        print(name, "emitted T/measurements:", actual, "clean work: 6")
        print(
            name,
            "pinned model T-equivalent/measurements:",
            (model_t, model_measurements),
        )
    print("All complex amplitudes restored; six work wires end in zero")


if __name__ == "__main__":
    main()
