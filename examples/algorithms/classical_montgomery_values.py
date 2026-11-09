"""Run python -m examples.algorithms.classical_montgomery_values.

Inspect numerical Montgomery history and bind the matching 21-wire append path.
"""

from collections import Counter
from typing import cast

import numpy as np

from examples.algorithms.register_windowed_mod_mul import (
    WindowedMontgomeryMultiplication,
)
from flagquantum.core.ir.program.register_types import (
    classical_montgomery_register_type,
    montgomery_register_type,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.classical_modular_multiplication import (
    controlled_modular_multiply_constant_classical_values,
    modular_double_classical_value,
)
from flagquantum.simulation.classical_montgomery_multiplication import (
    montgomery_multiply_classical_values,
    montgomery_window_classical_values,
)
from flagquantum.simulation.classical_montgomery_window import (
    montgomery_window_bits_classical_values,
)
from flagquantum.simulation.integer_encoding import (
    integer_from_bits,
    integer_to_bits,
    vectorized_integer_array_from_bits,
    vectorized_integer_array_to_bits,
)
from flagquantum.simulation.montgomery_values import (
    integer_to_montgomery_value,
    montgomery_inverse_value,
    montgomery_product_values,
    montgomery_to_integer_value,
    montgomery_value_domain,
)


def main() -> None:
    assert modular_double_classical_value(2, 3) == 1
    assert controlled_modular_multiply_constant_classical_values(1, 2, 2, 3) == (1, 1)
    quantum_word = montgomery_register_type(3, 3)
    classical_word = classical_montgomery_register_type(3, 3)
    assert quantum_word.num_bits == classical_word.num_bits == 3
    encoded = integer_to_montgomery_value(1, quantum_word.quantum_bits, 3)
    assert montgomery_to_integer_value(encoded, classical_word.classical_bits, 3) == 1
    assert (
        montgomery_product_values(
            encoded, montgomery_inverse_value(encoded, 3, 3), 3, 3
        )
        == encoded
    )
    assert tuple(montgomery_value_domain(3, 3)) == (1, 2)
    stored = np.array([[0, 1], [2, 7]])
    packed = vectorized_integer_array_to_bits(stored, classical_word.classical_bits)
    np.testing.assert_array_equal(
        vectorized_integer_array_from_bits(packed, classical_word.classical_bits),
        stored,
    )
    print(
        "Montgomery representation conversion/product/inverse and generic array round trip agree"
    )
    values = montgomery_multiply_classical_values(1, 2, 3, 1, 3)
    assert int(values["target"]) == 1 and int(values["qrom_indices"]) == 2
    assert int(values["reduced"]) == 0
    target, history = montgomery_window_classical_values(1, 2, 0, 0, 3, 1, 3)
    for offset in range(1, 3):
        target, history = montgomery_window_classical_values(
            1 >> offset, 2, target, history, 3, 1, 3
        )
    assert target == values["target"] and history == values["qrom_indices"]
    target_bits, bit_history = integer_to_bits(0, 4), 0
    for offset in range(3):
        step = montgomery_window_bits_classical_values(
            [(1 >> offset) & 1], 2, target_bits, 0, 3, 1, 3
        )
        target_bits = cast(list[int], step["target"])
        bit_history = (bit_history << 1) | cast(int, step["qrom_index"])
    assert integer_from_bits(target_bits, 4) == target
    assert bit_history == history
    print("Shared bit codecs and bit-array windows agree with the integer history")
    assert montgomery_multiply_classical_values(
        1,
        2,
        3,
        1,
        3,
        target=values["target"],
        history=values["qrom_indices"],
        reduced=values["reduced"],
        uncompute=True,
    ) == {"x": 1, "y": 2}
    print(
        "Numerical target/history/reduction:",
        int(target),
        int(history),
        int(values["reduced"]),
    )

    circuit = DynamicCircuit(21)
    operation = WindowedMontgomeryMultiplication()
    circuit.append(operation)
    counts = Counter(gate.name for gate in circuit.to_ir().instructions)
    emitted = counts["t"] + counts["tdg"], counts["measure"], circuit.n_qubits
    assert emitted == (128, 32, 21)
    print("Emitted T/measurement/declared wires:", emitted)
    print(
        "Pinned n=3,w=1,p=3 model T-equivalent/measurement/logical wires:",
        (128, 32, 21),
    )
    print("Forward target ports:", operation.target_wires)
    circuit.append(WindowedMontgomeryMultiplication(adjoint=True))
    print(
        "Forward and exact adjoint IR constructed; this example does not execute the circuit"
    )


if __name__ == "__main__":
    main()
