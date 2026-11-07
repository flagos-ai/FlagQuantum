"""Run python -m examples.algorithms.classical_signed_values.

Shared value codecs and coherent ones' complement conversion through append.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.signed_encoding import ones_complement_to_twos_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.bit_encoding import (
    bit_array_from_bits,
    bit_array_to_bits,
    bit_from_bits,
    bit_to_bits,
)
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits
from flagquantum.simulation.ones_complement_encoding import (
    ones_complement_array_from_bits,
    ones_complement_array_to_bits,
    ones_complement_from_bits,
    ones_complement_to_bits,
)


@dataclass(frozen=True)
class OnesToTwos:
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(5, self.instructions))


def main() -> None:
    assert bit_from_bits(bit_to_bits(True)) == 1
    flags = np.array([True, False])
    np.testing.assert_array_equal(bit_array_from_bits(bit_array_to_bits(flags)), flags)
    values = np.array([-3, -1, 0, 2, 3])
    np.testing.assert_array_equal(
        ones_complement_array_from_bits(ones_complement_array_to_bits(values, 3), 3),
        values,
    )
    weights = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(weights, 3 * weights - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word in range(8):
        bits = integer_to_bits(word, 3)
        value = ones_complement_from_bits(bits, 3)
        assert isinstance(value, int)
        if word == 7:
            # Preserve the established reversible full-word extension; scalar
            # negative-zero decode alone cannot define a bijective conversion.
            assert value == 0
            output = 4
        else:
            assert ones_complement_to_bits(value, 3) == bits
            output = integer_from_bits(integer_to_bits(value, 3, signed=True), 3)
        for spectator in (0, 1):
            initial[(word << 2) | spectator] = weights[2 * word + spectator]
            expected[(output << 2) | spectator] = weights[2 * word + spectator]
    gates = ones_complement_to_twos_ir((0, 1, 2), (3,))
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(OnesToTwos(gates))
    forward = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    inverse = ones_complement_to_twos_ir((0, 1, 2), (3,), adjoint=True)
    circuit.append(OnesToTwos(inverse))
    restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(gate.name for gate in gates)
    actual = counts["t"] + counts["tdg"], counts["measure"]
    assert actual == (4, 1)
    print("Emitted T/measurements/work:", (*actual, 1))
    print(
        "Pinned source abstract model: 1 Toffoli, 0 explicit measurements, 4 T equivalents"
    )
    print(
        "All complex amplitudes restored; one work wire zero; negative-zero extension retained"
    )


if __name__ == "__main__":
    main()
