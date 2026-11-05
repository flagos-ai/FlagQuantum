"""Use circuit.append(op) for coherent controlled register addition.

Run: python -m examples.algorithms.register_controlled_addition
Expected: each control polarity coherently selects 2+3=5, preserving the inactive
branch at 2 and returning all four quantum work wires to zero.
"""

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import torch

import flagquantum as fq
from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_constant_ir,
    controlled_add_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledAddition:
    """Bind explicit register work through the existing append protocol."""

    control_value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = controlled_add_ir(
            0,
            (1, 2),
            (3, 4, 5),
            (6, 7),
            (8, 9),
            (0, 1),
            (2, 3),
            control_value=self.control_value,
        )
        return circuit.compose(CircuitIR(10, gates))


@dataclass(frozen=True)
class ControlledConstantAddition:
    """Control only constant loading; the shared Add always executes."""

    control_value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = controlled_add_constant_ir(
            0,
            (1, 2, 3),
            -3,
            (4, 5, 6),
            (7, 8),
            (0, 1),
            control_value=self.control_value,
        )
        return circuit.compose(CircuitIR(9, gates))


def main() -> None:
    """Check both polarities, including the relative phase of control branches."""
    state = torch.zeros(1024, dtype=torch.complex128)
    state[((3 << 3) | 2) << 4] = 1 / sqrt(2)
    state[((1 << 5) | (3 << 3) | 2) << 4] = 1j / sqrt(2)
    for cv in (0, 1):
        circuit = DynamicCircuit(10, inputs=state, dtype=torch.complex128).append(
            ControlledAddition(cv)
        )
        expected = torch.zeros_like(state)
        for control, amp in ((0, 1 / sqrt(2)), (1, 1j / sqrt(2))):
            value = 5 if control == cv else 2
            expected[((control << 5) | (3 << 3) | value) << 4] = amp
        result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(32, -1),
            expected.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            f"Control={cv}: active b=5, inactive b=2; relative phase and all 4 clean work wires preserved"
        )
        print(
            "Measured gates:",
            dict(Counter(g.name for g in circuit.to_ir().instructions)),
        )

    state = torch.zeros(512, dtype=torch.complex128)
    state[2 << 5] = 1 / sqrt(2)
    state[((1 << 3) | 2) << 5] = 1j / sqrt(2)
    for cv in (0, 1):
        circuit = DynamicCircuit(9, inputs=state, dtype=torch.complex128).append(
            ControlledConstantAddition(cv)
        )
        expected = torch.zeros_like(state)
        for control, amp in ((0, 1 / sqrt(2)), (1, 1j / sqrt(2))):
            value = 7 if control == cv else 2
            expected[((control << 3) | value) << 5] = amp
        result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(32, -1),
            expected.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            f"Constant -3, control={cv}: active b=7, inactive b=2; phase preserved and all 5 work wires zero"
        )
        print(
            "Measured gates:",
            dict(Counter(g.name for g in circuit.to_ir().instructions)),
        )


if __name__ == "__main__":
    main()
