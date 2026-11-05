"""Coherent controlled modular arithmetic with both control polarities.

Run: python -m examples.algorithms.register_controlled_modular_arithmetic
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.controlled_modular_addition import (
    controlled_modular_add_ir,
)
from flagquantum.compiler.registers.controlled_modular_subtraction import (
    controlled_modular_subtract_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledModularArithmetic:
    """Bind a coherent control, three-bit residues, and nine clean work wires."""

    operation: Literal["add", "subtract"]
    control_value: int = 1

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        emit = {
            "add": controlled_modular_add_ir,
            "subtract": controlled_modular_subtract_ir,
        }[self.operation]
        gates = emit(
            0,
            (1, 2, 3),
            (4, 5, 6),
            3,
            tuple(range(7, 16)),
            tuple(range(6)),
            control_value=self.control_value,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    words = [(ctrl, a, b) for ctrl in (0, 1) for a in range(3) for b in range(3)]
    coeff = torch.arange(len(words), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 16, dtype=torch.complex128)
    for (ctrl, a, b), amp in zip(words, amps, strict=True):
        state[((ctrl << 6) | (a << 3) | b) << 9] = amp
    operations: tuple[Literal["add", "subtract"], ...] = ("add", "subtract")
    for cv in (0, 1):
        for operation in operations:
            expected = torch.zeros_like(state)
            for (ctrl, a, b), amp in zip(words, amps, strict=True):
                value = (b + (a if operation == "add" else -a)) % 3 if ctrl == cv else b
                expected[((ctrl << 6) | (a << 3) | value) << 9] = amp
            circuit = DynamicCircuit(16, inputs=state, dtype=torch.complex128).append(
                ControlledModularArithmetic(operation, cv)
            )
            result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(8, -1),
                expected.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                operation,
                "control value",
                cv,
                ": coherent amplitudes correct; nine work wires zero",
            )
            print(
                "Measured primitive gates:",
                dict(Counter(g.name for g in circuit.to_ir().instructions)),
            )
            if operation == "add":
                restored = run_dynamic(
                    circuit.append(ControlledModularArithmetic("subtract", cv)),
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
    print("Both controlled addition/subtraction round trips verified.")
    print(
        "Independent source model: add 64 T / 16 measurements; subtract 80 T / 20 measurements."
    )
    print("Work budget: nine quantum wires, six reused classical slots.")


if __name__ == "__main__":
    main()
