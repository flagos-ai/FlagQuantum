"""Run python -m examples.algorithms.classical_finite_field_values.

Install the optional galois numerical dependency before running this example.
"""

from collections import Counter

import numpy as np
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    bind_field_outputs,
    gf2_multiplication,
)
from flagquantum.core.ir.program.register_compatibility import (
    check_register_types_consistent,
)
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding


def main() -> None:
    """Bind the numerical field's explicit binary modulus to existing gates."""
    field = FiniteFieldEncoding(2, 3, [3, 1, 0])
    quantum_word = field.register_type()
    classical_word = field.register_type(classical=True)
    assert quantum_word.num_qubits == classical_word.num_cbits == 3
    assert check_register_types_consistent(
        quantum_word, fixed_point_register_type(3, 0)
    )
    assert field.order == 8 and field.bitsize == 3
    bits = np.array(
        [field.to_bits(value) for value in field.get_domain()], dtype=np.uint8
    )
    np.testing.assert_array_equal(field.from_bits_array(bits), field.gf_type.elements)
    product = field.gf_type(3) * field.gf_type(5)
    assert int(product) == 4 and field.to_bits(product) == [1, 0, 0]

    width = int(quantum_word.num_qubits)
    modulus = int(dict(quantum_word.parameters)["polynomial"])
    x, y, target = (tuple(range(i * width, (i + 1) * width)) for i in range(3))
    registers = (x, y, target)
    plan = gf2_multiplication(*registers, (), modulus=modulus, algorithm="quadratic")
    gates = bind_field_outputs(plan, registers)
    state = torch.zeros(512, dtype=torch.complex128)
    state[(3 << 6) | (5 << 3)] = 1
    result = (
        fq.Circuit(9, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(9, gates))
        .state()
        .flatten()
    )
    expected = torch.zeros_like(state)
    expected[(3 << 6) | (5 << 3) | int(product)] = 1
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)
    print("GF(8) numerical and appended quantum product agree for 3 * 5 = 4")
    print(
        "Emitted decomposition primitives",
        dict(Counter(g.name for g in plan.instructions)),
    )
    print("Emitted output-binding SWAPs", len(gates) - len(plan.instructions))
    print("Allocated wires", 9, "including", 3, "product outputs and", 0, "work wires")
    print("Model estimates are reported separately in the pinned source regression")


if __name__ == "__main__":
    main()
