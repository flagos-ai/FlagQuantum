"""Run python -m examples.algorithms.classical_register_values."""

from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.bitwise import xor_constant_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.classical_bitwise import xor_constant_classical_value
from flagquantum.simulation.classical_registers import (
    bind_integer_register_values,
    collect_integer_register_value,
)


@dataclass(frozen=True)
class XorEachWord:
    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = tuple(
            g for start in (0, 2, 4, 6) for g in xor_constant_ir((start, start + 1), 1)
        )
        return circuit.compose(CircuitIR(8, gates))


def main() -> None:
    words = np.array([[1, 2], [3, 0]], dtype=np.int64)
    bound = bind_integer_register_values("x", {"x": words}, 2, shape=(2, 2))
    assert tuple(bound) == ((0, 0), (0, 1), (1, 0), (1, 1))
    mapped = 0
    initial_word = 0
    output_values = {}
    for idx, value in bound.items():
        assert isinstance(value, (int, np.integer))
        initial_word = (initial_word << 2) | int(value)
        output_values[idx] = xor_constant_classical_value(value, 1)
    output = collect_integer_register_value(output_values, 2, shape=(2, 2))
    assert isinstance(output, np.ndarray) and output.dtype == np.uint8
    for value in output.flat:
        mapped = (mapped << 2) | int(value)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[initial_word] = 1
    circuit = fq.Circuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(XorEachWord())
    expected = torch.zeros_like(initial)
    expected[mapped] = 1
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    print("Row-major words [[1,2],[3,0]] XOR 1 -> [[0,3],[2,1]]")
    print("circuit.append(op) matches; four X, no work, measurements or T gates")


if __name__ == "__main__":
    main()
