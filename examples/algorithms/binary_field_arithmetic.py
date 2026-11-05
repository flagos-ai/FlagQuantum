"""Run polynomial-basis field operations through circuit.append(op).

Run: python -m examples.algorithms.binary_field_arithmetic
No reference packages are needed to execute this example.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.field_multiplication import (
    bind_field_outputs,
    gf2_multiplication,
)
from flagquantum.core.ir import CircuitIR, Instruction


@dataclass(frozen=True)
class FieldOperation:
    """Bind an internal register decomposition to the existing append protocol."""

    n_qubits: int
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(self.n_qubits, self.instructions))


def main() -> None:
    """Compute 3 * 5 = 4 in GF(8), modulo x^3 + x + 1."""
    x, y, target = (0, 1, 2), (3, 4, 5), (6, 7, 8)
    plan = gf2_multiplication(x, y, target, (), modulus=0b1011)
    instructions = bind_field_outputs(plan, (x, y, target))
    state = torch.zeros(512, dtype=torch.complex128)
    state[(3 << 6) | (5 << 3)] = 1
    circuit = fq.Circuit(9, inputs=state, dtype=torch.complex128)
    result = circuit.append(FieldOperation(9, instructions)).state().flatten()
    assert int(result.abs().argmax()) == (3 << 6) | (5 << 3) | 4
    print("GF(8): 3 * 5 = 4; source registers preserved")
    print("Emitted gates:", dict(Counter(g.name for g in instructions)))
    print("Output binding SWAPs:", len(instructions) - len(plan.instructions))


if __name__ == "__main__":
    main()
