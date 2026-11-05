"""Modulo-three register arithmetic and exact source-adjoint restoration.

Run: python -m examples.algorithms.register_modular_arithmetic
Expected: all coherent valid-residue amplitudes correct, eight work wires zero,
modular addition followed by literal source modular subtraction restoring input.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular_addition import modular_add_ir
from flagquantum.compiler.registers.modular_subtraction import modular_subtract_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularArithmetic:
    """Bind two three-bit residue words and eight explicitly zero work wires."""

    operation: Literal["add", "subtract"]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        emit = {"add": modular_add_ir, "subtract": modular_subtract_ir}[self.operation]
        gates = emit((0, 1, 2), (3, 4, 5), 3, tuple(range(6, 14)), (0, 1, 2))
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    pairs = [(a, b) for a in range(3) for b in range(3)]
    coeff = torch.arange(len(pairs), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 14, dtype=torch.complex128)
    for (a, b), amp in zip(pairs, amps, strict=True):
        state[((a << 3) | b) << 8] = amp
    counts: dict[str, Counter[str]] = {}
    operations: tuple[Literal["add", "subtract"], ...] = ("add", "subtract")
    for operation in operations:
        expected = torch.zeros_like(state)
        for (a, b), amp in zip(pairs, amps, strict=True):
            value = (a + b if operation == "add" else b - a) % 3
            expected[((a << 3) | value) << 8] = amp
        circuit = DynamicCircuit(14, inputs=state, dtype=torch.complex128).append(
            ModularArithmetic(operation)
        )
        actual = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            actual.final_states.reshape(8, -1),
            expected.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        counts[operation] = Counter(g.name for g in circuit.to_ir().instructions)
        if operation == "add":
            restored = run_dynamic(
                circuit.append(ModularArithmetic("subtract")),
                shots=8,
                seed=41,
                strategy="trajectory",
            )
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                state.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
        print(
            operation,
            ": all valid-residue complex amplitudes correct; eight work wires zero",
        )
        print("Measured gates:", dict(counts[operation]))
    print(
        "Addition/subtraction round trip verified. Model: add 44 T/11 measurements; subtract 60 T/15 measurements."
    )
    print(
        "Work budget: eight quantum wires, three reused classical slots; input states must satisfy residue domains."
    )


if __name__ == "__main__":
    main()
