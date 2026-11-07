"""Bind an inferred XOR table and compare its resource model with emitted gates.

Run python -m examples.algorithms.register_qrom_table.
"""

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.qrom_table import QromTable
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    table = QromTable.from_shape(4, 2).with_data([1, 3, 0, 2])
    gates = table.bind((), ((0, 1),), (((2, 3),),), (4,), (0,))
    initial = torch.zeros(32, dtype=torch.complex128)
    amplitudes = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex128) / 2
    initial[::8] = amplitudes
    expected = torch.zeros_like(initial)
    for row, value in enumerate((1, 3, 0, 2)):
        expected[(row * 4 + value) * 2] = amplitudes[row]
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, gates))
    result = run_dynamic(circuit, shots=16, seed=73, strategy="batched")
    torch.testing.assert_close(
        result.final_states, expected.expand_as(result.final_states), atol=2e-12, rtol=0
    )
    report = estimate_ftqc_resources(table, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    print("Declared logical categories:", report.logical.gates)
    print("Declared peak logical qubits:", report.logical.peak_logical_qubits)
    print("Emitted primitive T/Tdag:", report.emitted.t_count)
    print("Caller-owned work wires:", table.workspace_size)
    print("Complex amplitudes and zero work verified for sampled measurement histories")


if __name__ == "__main__":
    main()
