"""Run both subtraction directions with circuit.append(op).

Run: python -m examples.algorithms.register_subtraction
Expected: signed two-bit -1 minus three-bit 2 produces 5 (signed -3),
then a second subtraction restores 2; three-bit 2 -= 3 produces 7.
No upstream reference packages are required.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic.subtraction_component import (
    subtract_from_program,
    subtraction_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class RegisterSubtraction:
    """Bind an arithmetic decomposition to the existing append protocol."""

    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(circuit.n_qubits, self.instructions))


def verify(circuit: DynamicCircuit, index: int) -> None:
    """Check each sampled complex output, including every clean work wire."""
    result = run_dynamic(circuit, shots=16, seed=31, strategy="trajectory")
    expected = torch.zeros(1 << circuit.n_qubits, dtype=torch.complex128)
    expected[index] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )


def main() -> None:
    """Keep subtraction direction and signed extension explicit."""
    graph = subtraction_program(RegisterType("signed", 2), RegisterType("signed", 3))
    gates = bind_register_program(
        graph, ((0, 1), (2, 3, 4)), clean=(5, 6, 7)
    ).circuit.instructions
    state = torch.zeros(256, dtype=torch.complex128)
    state[((3 << 3) | 2) << 3] = 1
    operation = RegisterSubtraction(gates)
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(operation)
    verify(circuit, ((3 << 3) | 5) << 3)
    circuit.append(operation)
    verify(circuit, ((3 << 3) | 2) << 3)
    print("Signed a-b: -1 - 2 = -3 (three-bit pattern 5); all 3 work wires zero")
    print("A second a-b operation restores the original b")
    print("Measured gates:", dict(Counter(g.name for g in gates)))

    graph = subtract_from_program(RegisterType("uint", 3))
    gates = bind_register_program(
        graph, ((0, 1, 2), (3, 4, 5)), clean=(6, 7)
    ).circuit.instructions
    state = torch.zeros(256, dtype=torch.complex128)
    state[((3 << 3) | 2) << 2] = 1
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(RegisterSubtraction(gates))
    verify(circuit, ((3 << 3) | 7) << 2)
    print("Unsigned b-=a: 2 - 3 mod 8 = 7; both carry wires zero")
    print("Measured gates:", dict(Counter(g.name for g in gates)))


if __name__ == "__main__":
    main()
