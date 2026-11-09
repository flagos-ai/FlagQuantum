"""Bind an inferred XOR table and compare its resource model with emitted gates.

Run python -m examples.algorithms.register_qrom_table.
Add --symbolic to query the unknown-data model without circuit emission.
"""

import argparse

import numpy as np
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.qrom_table import QromTable
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import (
    estimate_ftqc_resources,
    estimate_symbolic_ftqc_resources,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_lookup import lookup_classical_values


def main(*, symbolic: bool = False) -> None:
    if symbolic:
        from sympy import symbols

        rows, bits, controls = symbols("N b c", integer=True, positive=True)
        table = QromTable.from_shape(rows, bits, num_controls=controls)
        assert (
            table.signature.n_qubits() == controls + table.selection_bitsizes[0] + bits
        )
        formula_report = estimate_symbolic_ftqc_resources(table.resource_formula())
        print("Unknown-data logical formula:", formula_report.logical)
        print("Model T equivalents:", formula_report.t_equivalents)
        print("No lookup values or circuit emission; dimensions remain symbolic")
        return
    data = (np.array([1, 3, 0, 2], dtype=int),)
    table = QromTable.from_shape(4, 2).with_data(*data)
    assert table.signature.n_qubits() == 4
    assert [register.name for register in table.signature] == ["selection", "target0_"]
    gates = table.bind_packed((0, 1, 2, 3), (4,), (0,))
    initial = torch.zeros(32, dtype=torch.complex128)
    amplitudes = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex128) / 2
    initial[::8] = amplitudes
    expected = torch.zeros_like(initial)
    for row in range(4):
        value = lookup_classical_values(data, (4,), {"selection": row})["target0_"]
        expected[(row * 4 + int(value)) * 2] = amplitudes[row]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic", action="store_true")
    main(symbolic=parser.parse_args().symbolic)
