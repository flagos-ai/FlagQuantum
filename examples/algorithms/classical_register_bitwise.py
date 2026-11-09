"""Run python -m examples.algorithms.classical_register_bitwise."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.bits.bitwise import bitwise_not_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.classical_bitwise import (
    bitwise_not_classical_value,
    xor_classical_values,
    xor_constant_classical_value,
)


@dataclass(frozen=True)
class ComplementWord:
    width: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(
            CircuitIR(self.width, bitwise_not_ir(tuple(range(self.width))))
        )


def main() -> None:
    assert xor_classical_values(5, 3) == (5, 6)
    assert xor_constant_classical_value(5, 3) == 6
    unsigned = bitwise_not_classical_value(6, 3, unsigned=True)
    signed = bitwise_not_classical_value(-2, 3, unsigned=False)
    assert unsigned == signed == 1
    # Raw word 110 means unsigned 6 or signed -2; both circuits output 001.
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[6] = 1
    circuit = fq.Circuit(3, inputs=initial, dtype=torch.complex128)
    circuit.append(ComplementWord(3))
    expected = torch.zeros_like(initial)
    expected[1] = 1
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    print("Classical XOR: 5 XOR 3 = 6")
    print("Three-bit complement: unsigned 6 or signed -2 -> stored word 001")
    print(
        "circuit.append(op) matches the classical output; 3 X, no work or measurements"
    )


if __name__ == "__main__":
    main()
