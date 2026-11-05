"""Run coherent constant/scale modular additions through circuit.append(op).

Run: python -m examples.algorithms.register_modular_scaled_addition
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular_constant_addition import (
    controlled_modular_add_constant_ir,
)
from flagquantum.compiler.registers.modular_scaled_addition import (
    controlled_modular_scaled_add_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularAccumulation:
    """Bind two-bit words, one control, and explicit clean work at modulus three."""

    operation: Literal["constant", "scaled"]
    value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        if self.operation == "constant":
            gates = controlled_modular_add_constant_ir(
                0, (3, 4), self.value, 3, tuple(range(6, 16)), (0, 1, 2, 3)
            )
        else:
            gates = controlled_modular_scaled_add_ir(
                0, (1, 2), (3, 4), self.value, 3, tuple(range(5, 16)), (0, 1, 2, 3)
            )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    labels = [(c, x, y) for c in (0, 1) for x in range(4) for y in range(3)]
    coeff = torch.arange(len(labels), dtype=torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    inputs = torch.zeros(1 << 16, dtype=torch.complex128)
    for (c, x, y), amp in zip(labels, amplitudes, strict=True):
        inputs[((c << 4) | (x << 2) | y) << 11] = amp
    operations: tuple[Literal["constant", "scaled"], ...] = ("constant", "scaled")
    for operation in operations:
        value = 1 if operation == "constant" else -1
        expected = torch.zeros_like(inputs)
        for (c, x, y), amp in zip(labels, amplitudes, strict=True):
            delta = value * (x if operation == "scaled" else 1)
            result = (y + delta) % 3 if c else y
            expected[((c << 4) | (x << 2) | result) << 11] = amp
        circuit = DynamicCircuit(16, inputs=inputs, dtype=torch.complex128).append(
            ModularAccumulation(operation, value)
        )
        output = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            output.final_states.reshape(8, -1),
            expected.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        inverse = (-value) % 3 if operation == "constant" else -value
        restored = run_dynamic(
            circuit.append(ModularAccumulation(operation, inverse)),
            shots=8,
            seed=41,
            strategy="trajectory",
        )
        torch.testing.assert_close(
            restored.final_states.reshape(8, -1),
            inputs.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            operation,
            ": all coherent complex amplitudes and inverse restoration correct; work zero",
        )
        print(
            "Measured primitives:",
            dict(counts),
        )
    print("Independent source T/measurement estimates: constant 44/11; scaled 96/24.")
    print(
        "Constant work: ten wires; scaled work: eleven wires; four reused classical slots."
    )


if __name__ == "__main__":
    main()
