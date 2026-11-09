"""Compare the pinned linear and tree greater-than paths via circuit.append(op).

Run: python -m examples.algorithms.register_greater
Expected: both paths preserve every complex input/target amplitude and clean
all work. Counts report the two source decompositions separately. No references.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.comparison.less_equal import less_equal_requirements
from flagquantum.compiler.registers.comparison.linear_comparison import (
    linear_greater_ir,
    linear_greater_requirements,
)
from flagquantum.compiler.registers.comparison.ordering import greater_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class Greater:
    """Bind one of two explicitly selected three-bit source decompositions."""

    route: Literal["linear", "tree"]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        if self.route == "linear":
            q, c = linear_greater_requirements(3)
            gates = linear_greater_ir(
                (0, 1, 2), (3, 4, 5), 6, tuple(range(7, 7 + q)), tuple(range(c))
            )
        elif self.route == "tree":
            q, c = less_equal_requirements(3, 3)
            gates = greater_ir(
                (0, 1, 2), (3, 4, 5), 6, tuple(range(7, 7 + q)), tuple(range(c))
            )
        else:
            raise ValueError("Unknown comparison route")
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    data = torch.arange(128, dtype=torch.int64)
    pair = data >> 1
    target = data & 1
    mapped = (pair << 1) | (target ^ ((pair >> 3) > (pair & 7)))
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for route in ("linear", "tree"):
        q, c = (
            linear_greater_requirements(3)
            if route == "linear"
            else less_equal_requirements(3, 3)
        )
        n_qubits = 7 + q
        state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
        state[data << q] = amps
        expected = torch.zeros_like(state)
        expected[mapped << q] = amps
        operation = Greater(route)
        circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
            operation
        )
        result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(32, -1),
            expected.expand(32, -1),
            rtol=0,
            atol=1e-12,
        )
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        restored = run_dynamic(
            circuit.append(operation), shots=8, seed=41, strategy="trajectory"
        )
        torch.testing.assert_close(
            restored.final_states.reshape(8, -1),
            state.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            f"{route}: complete unsigned a>b amplitudes correct; all work zero; double XOR restores input"
        )
        print(
            "Quantum work:", q, "Measurement slots:", c, "Measured gates:", dict(counts)
        )


if __name__ == "__main__":
    main()
