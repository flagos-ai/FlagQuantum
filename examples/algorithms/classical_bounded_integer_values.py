"""Run python -m examples.algorithms.classical_bounded_integer_values."""

from itertools import product

import numpy as np
import torch

from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.bounded_integer_encoding import (
    bounded_integer_array_from_bits,
    bounded_integer_array_to_bits,
    bounded_integer_from_bits,
    bounded_integer_to_bits,
)
from flagquantum.simulation.bounded_integer_values import bounded_integer_domain


def main() -> None:
    values = np.array(list(bounded_integer_domain(2, 3)))
    bits = bounded_integer_array_to_bits(values, 2, 3)
    np.testing.assert_array_equal(bounded_integer_array_from_bits(bits, 2, 3), values)
    print(
        "Bounded addresses and big-endian words:",
        list(zip(values.tolist(), bits.tolist(), strict=True)),
    )

    grid = torch.arange(1, 7, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for address, target in product(bounded_integer_domain(2, 3), (0, 1)):
        stored = bounded_integer_from_bits(bounded_integer_to_bits(address, 2, 3), 2, 3)
        amplitude = weights[2 * address + target]
        initial[(stored << 2) | (target << 1)] = amplitude
        output = target ^ (address != 2)
        phase = (
            (1j if target == 0 else -1j)
            if address == 1
            else (-1 if address == 2 and target else 1)
        )
        expected[(stored << 2) | (output << 1)] = phase * amplitude
    op = SelectPauliLCU(2, 1, ("X", "Y", "Z")).make_bounded_on(
        selection=(0, 1), target=(2,), workspace=(3,), classical_bits=(0,)
    )
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    print("Model T equivalents:", report.logical.gates.total_t_count())
    print("Model measurements:", report.logical.gates.measurement)
    print("Model peak logical qubits:", report.logical.peak_logical_qubits)
    print("Emitted explicit T:", report.emitted.t_count)
    print("Emitted measurements:", report.emitted.unconditional.get("measure", 0))
    print("Work wires: 1, returned to zero; all complex amplitudes agree")


if __name__ == "__main__":
    main()
