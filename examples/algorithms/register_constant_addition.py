"""Add a constant using circuit.append(op), without reference dependencies.

Run: python -m examples.algorithms.register_constant_addition
Expected: adding -3 to three bits maps |2> to |7>, with all five work wires zero.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic import add_constant_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ConstantAddition:
    """Bind the shared integer decomposition to the existing append protocol."""

    value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = add_constant_ir((0, 1, 2), self.value, (3, 4, 5), (6, 7), (0, 1))
        return circuit.compose(CircuitIR(8, gates))


def main() -> None:
    """Check every sampled branch, including the complete clean workspace."""
    state = torch.zeros(256, dtype=torch.complex128)
    state[2 << 5] = 1
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128).append(
        ConstantAddition(-3)
    )
    result = run_dynamic(circuit, shots=16, seed=31, strategy="trajectory")
    expected = torch.zeros_like(state)
    expected[7 << 5] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    print("3-bit unsigned addition: 2 + (-3) mod 8 = 7; all 5 work wires are zero")
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    print("Measured gate counts:", dict(counts))
    print("T/TDG total:", counts["t"] + counts["tdg"])
    print("Measurements:", counts["measure"])


if __name__ == "__main__":
    main()
