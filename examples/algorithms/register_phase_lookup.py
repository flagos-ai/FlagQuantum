"""Use controlled phase-gradient lookup through circuit.append(op).

Run python -m examples.algorithms.register_phase_lookup.
"""

from cmath import exp
from collections import Counter
from math import sqrt, tau

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.phase.phase_gradient_lookup import (
    phase_gradient_lookup_ir,
    phase_gradient_lookup_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    words, width = (1, 3, 0, 5), 3
    work_size, slot_count = phase_gradient_lookup_requirements(1, 2, width, words)
    work = tuple(range(6, 6 + work_size))
    gates = phase_gradient_lookup_ir(
        (0,), (1, 2), (3, 4, 5), words, work, tuple(range(slot_count))
    )
    qubits = 6 + work_size
    initial = torch.zeros(1 << qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    weights = torch.arange(1, 9, dtype=torch.float64)
    coefficients = torch.complex(weights, weights % 3 - 1)
    coefficients /= coefficients.norm()
    for control in (0, 1):
        for row, word in enumerate(words):
            phase = exp(1j * tau * word / (1 << width)) if control else 1
            for gradient in range(1 << width):
                index = (((control << 2 | row) << width) | gradient) << work_size
                amplitude = (
                    coefficients[control * 4 + row]
                    * exp(-1j * tau * gradient / (1 << width))
                    / sqrt(1 << width)
                )
                initial[index] = amplitude
                expected[index] = amplitude * phase
    circuit = DynamicCircuit(qubits, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(qubits, gates))
    result = run_dynamic(circuit, shots=8, seed=127, strategy="batched")
    torch.testing.assert_close(
        result.final_states, expected.expand_as(result.final_states), atol=3e-12, rtol=0
    )
    table = QromTable.from_shape(4, width, num_controls=1).with_data(words)
    print("One QROM child logical categories:", table.resource_model().gates)
    print("Full emitted primitive counts:", dict(Counter(g.name for g in gates)))
    print(
        "Clean work including angle:", work_size, "reused classical slots:", slot_count
    )
    print(
        "Complex phase, coherent inactive controls, unchanged gradient and zero work verified"
    )


if __name__ == "__main__":
    main()
