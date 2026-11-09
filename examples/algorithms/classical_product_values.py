"""Run python -m examples.algorithms.classical_product_values.

Compare classical square values and coherent append execution without Qualtran.
"""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic.integer_square import integer_square_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_product_accumulation import (
    accumulate_product_classical_values,
)
from flagquantum.simulation.classical_square import square_classical_values


@dataclass(frozen=True)
class SquareOperation:
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(8, self.instructions))


def main() -> None:
    assert accumulate_product_classical_values(3, 2, 15, 4) == (3, 2, 5)
    assert accumulate_product_classical_values(3, 2, 5, 4, adjoint=True) == (3, 2, 15)
    squared = square_classical_values({"a": 3})
    assert squared == {"a": 3, "result": 9}
    assert square_classical_values(squared, uncompute=True) == {"a": 3}
    gates = integer_square_ir((0, 1), (2, 3, 4, 5), (6, 7))
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[0] = 0.8
    initial[3 << 6] = torch.tensor(0.6j, dtype=torch.complex128)
    expected = initial.clone()
    expected[3 << 6] = 0
    result = squared["result"]
    assert isinstance(result, int)
    expected[(3 << 6) | (result << 2)] = initial[3 << 6]
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    operation = SquareOperation(gates)
    circuit.append(operation)
    computed = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
    torch.testing.assert_close(
        computed.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    circuit.append(operation)
    erased = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
    torch.testing.assert_close(
        erased.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print(
        "3*2+15 mod16 = 5; inverse restores15; square3 = 9 and checked erasure succeeds"
    )
    print("Append square: 8 emitted T, 2 measurements, 2 clean work wires")
    print("Source square model: 2 Toffoli; 8 T equivalents under the 4-T assumption")


if __name__ == "__main__":
    main()
