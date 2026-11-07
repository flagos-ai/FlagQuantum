"""Run python -m examples.algorithms.classical_comparison_values.

Source numerical predicates and existing measured equality through append.
"""

from collections import Counter

import torch

from examples.algorithms.register_equality import Equality
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_comparison import (
    equal_classical_values,
    less_equal_classical_values,
    less_than_constant_classical_values,
)
from flagquantum.simulation.classical_greater_comparison import (
    controlled_linear_greater_classical_values,
    linear_greater_classical_values,
)
from flagquantum.simulation.classical_half_comparison import (
    half_comparison_classical_values,
)


def main() -> None:
    assert less_than_constant_classical_values(2, 0, 3) == (2, 1)
    assert less_equal_classical_values(2, 3, 1) == (2, 3, 0)
    assert equal_classical_values(2, 2, 0) == (2, 2, 1)
    assert linear_greater_classical_values(3, 1, 0) == (3, 1, 1)
    assert controlled_linear_greater_classical_values(0, 3, 1, 0) == (0, 3, 1, 0)
    half = half_comparison_classical_values(3, 1, 3)
    assert half == {"a": 3, "b": 1, "c": 14, "target": 1}
    assert half_comparison_classical_values(
        3, 1, 3, uncompute=True, difference=half["c"], target=half["target"]
    ) == {"a": 3, "b": 1}
    print("Numerical predicates and checked half-comparison logical erasure agree")

    grid = torch.arange(1, 129, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data in range(128):
        left, right, target = data >> 4, (data >> 1) & 7, data & 1
        updated = equal_classical_values(left, right, target)[-1]
        initial[data << 2] = weights[data]
        expected[((data & ~1) | updated) << 2] = weights[data]
    operation = Equality()
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128)
    circuit.append(operation)
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    emitted = counts["t"] + counts["tdg"], counts["measure"]
    assert emitted == (8, 2)
    circuit.append(operation)
    restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print("Emitted equality T/measurements/work:", (*emitted, 2))
    print("Pinned source model: 8 T equivalents and 2 measurements (separate budget)")
    print(
        "Every complex amplitude agrees; two work wires zero; double equality restores input"
    )


if __name__ == "__main__":
    main()
