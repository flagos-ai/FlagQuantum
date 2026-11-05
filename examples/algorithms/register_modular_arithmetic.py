"""Modulo-three register arithmetic and exact source-adjoint restoration.

Run: python -m examples.algorithms.register_modular_arithmetic
Expected: all coherent valid-residue amplitudes correct, eight work wires zero,
each original operation followed by its exact component adjoint restores input.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular_addition import (
    modular_add_adjoint_ir,
    modular_add_ir,
)
from flagquantum.compiler.registers.modular_subtraction import (
    modular_subtract_adjoint_ir,
    modular_subtract_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularArithmetic:
    """Bind two three-bit residue words and eight explicitly zero work wires."""

    operation: Literal["add", "subtract"]
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        emit = {
            ("add", False): modular_add_ir,
            ("add", True): modular_add_adjoint_ir,
            ("subtract", False): modular_subtract_ir,
            ("subtract", True): modular_subtract_adjoint_ir,
        }[(self.operation, self.inverse)]
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
    operations: tuple[Literal["add", "subtract"], ...] = ("add", "subtract")
    for operation in operations:
        for inverse in (False, True):
            expected = torch.zeros_like(state)
            for (a, b), amp in zip(pairs, amps, strict=True):
                add = (operation == "add") != inverse
                value = (b + a if add else b - a) % 3
                expected[((a << 3) | value) << 8] = amp
            circuit = DynamicCircuit(14, inputs=state, dtype=torch.complex128)
            circuit.append(ModularArithmetic(operation, inverse))
            actual = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                actual.final_states.reshape(8, -1),
                expected.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            circuit.append(ModularArithmetic(operation, not inverse))
            restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                state.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                operation,
                "inverse",
                inverse,
                ": complex amplitudes correct; work zero; original inverse restored input",
            )
            print("Measured primitives:", dict(counts))
    print(
        "Independent source estimates per direction: ModAdd 44 T/11 measurements; ModSub 60 T/15 measurements."
    )
    print(
        "Work budget: eight quantum wires, three reused classical slots; input states must satisfy residue domains."
    )


if __name__ == "__main__":
    main()
