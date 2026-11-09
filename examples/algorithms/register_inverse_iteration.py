"""Append a complete two-bit inverse iteration and its original reverse.

Run: python -m examples.algorithms.register_inverse_iteration
Uses 22 qubits (64 MiB per state) and a single trajectory per check.
"""

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.inversion.inverse_iteration import (
    inverse_iteration_ir,
    inverse_iteration_outputs,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class InverseIteration:
    """Bind four two-bit words, three logical flags and eleven clean work wires."""

    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        r: tuple[int, ...] = (4, 5)
        work = tuple(range(11, 22))
        if self.adjoint:
            r, work = inverse_iteration_outputs(r, work)
        gates = inverse_iteration_ir(
            (0, 1),
            (2, 3),
            r,
            (6, 7),
            8,
            9,
            10,
            3,
            work,
            (0, 1, 2),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def pack(values: tuple[int, ...], ports: tuple[tuple[int, ...], ...]) -> int:
    raw = 0
    for value, wires in zip(values, ports, strict=True):
        for i, wire in enumerate(wires):
            raw |= ((value >> (len(wires) - 1 - i)) & 1) << (21 - wire)
    return raw


def main() -> None:
    """Check a complex superposition including initial zero-input termination."""
    u, v, r, s = (0, 1), (2, 3), (4, 5), (6, 7)
    work = tuple(range(11, 22))
    rr, ww = inverse_iteration_outputs(r, work)
    ports = (u, v, r, s, (8,), (9,), (10,))
    outputs = (u, v, rr, s, (8,), (9,), (10,))
    state = torch.zeros(1 << 22, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    # Source algorithm-reachable states; marker and terminal are fresh zero.
    cases = (
        ((3, 0, 0, 1, 0, 1, 0), (3, 0, 0, 1, 1, 0, 1), 1j / sqrt(3)),
        ((3, 1, 0, 1, 0, 1, 0), (1, 1, 1, 2, 1, 1, 0), 1 / sqrt(3)),
        ((3, 2, 0, 1, 0, 1, 0), (3, 1, 0, 1, 1, 1, 0), -1j / sqrt(3)),
    )
    for before, after, amp in cases:
        state[pack(before, ports)] = torch.tensor(amp, dtype=torch.complex128)
        expected[pack(after, outputs)] = torch.tensor(amp, dtype=torch.complex128)
    circuit = DynamicCircuit(22, inputs=state, dtype=torch.complex128)
    circuit.append(InverseIteration())
    result = run_dynamic(circuit, shots=1, seed=71, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(-1), expected, rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(InverseIteration(adjoint=True))
    restored = run_dynamic(circuit, shots=1, seed=72, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(-1), state, rtol=0, atol=1e-12
    )
    print("Complete complex forward/reverse vectors and returned work zero verified.")
    print("Physical coefficient output:", rr, "clean work:", ww)
    print("Measured primitives:", dict(counts))
    print(
        "Source models: default 120 T; exact seven-T swaps 147 T; 21 measurements; 22 qubits."
    )


if __name__ == "__main__":
    main()
