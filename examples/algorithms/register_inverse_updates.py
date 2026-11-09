"""Append nontrivial modular inverse scaling, retaining original output roles.

Run: python -m examples.algorithms.register_inverse_updates
This explicit statevector example uses 21 qubits (32 MiB per state) and one shot.
"""

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.inversion.inverse_arithmetic import (
    inverse_arithmetic_ir,
)
from flagquantum.compiler.registers.modular.inversion.inverse_scaling import (
    inverse_scaling_ir,
    inverse_scaling_outputs,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class CoefficientScaling:
    """Bind two-bit words, four logical flags and nine clean work wires."""

    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        r: tuple[int, ...] = (4, 5)
        work = tuple(range(12, 21))
        if self.adjoint:
            r, work = inverse_scaling_outputs(r, work)
        gates = inverse_scaling_ir(
            (0, 1),
            (2, 3),
            r,
            (6, 7),
            8,
            9,
            10,
            11,
            3,
            work,
            (0, 1, 2),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


@dataclass(frozen=True)
class ConditionalUpdates:
    """Bind two one-bit word pairs and two clean conditional-arithmetic wires."""

    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = inverse_arithmetic_ir(
            (0,), (1,), (2,), (3,), 4, 5, (6, 7), (0,), adjoint=self.adjoint
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def pack(
    values: tuple[int, ...], ports: tuple[tuple[int, ...], ...], width: int
) -> int:
    out = 0
    for value, wires in zip(values, ports, strict=True):
        for i, wire in enumerate(wires):
            out |= (value >> (len(wires) - 1 - i) & 1) << (width - 1 - wire)
    return out


def arithmetic_example() -> None:
    v = torch.arange(64, dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(256, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for raw, amp in enumerate(amps):
        u, value, r, s, branch, active = tuple(raw >> i & 1 for i in reversed(range(6)))
        if active and not branch:
            value ^= u
            s ^= r
        out = (u << 5) | (value << 4) | (r << 3) | (s << 2) | (branch << 1) | active
        state[raw << 2], expected[out << 2] = amp, amp
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(ConditionalUpdates())
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(ConditionalUpdates(adjoint=True))
    restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=1e-12
    )
    print(
        "Conditional ring arithmetic: complete complex amplitudes, two clean work bits, inverse restoration."
    )
    print("Measured arithmetic primitives:", dict(counts))
    print(
        "Independent source estimates: 12 T, three measurements, eight qubits; one-bit Clifford estimate has a known source exception."
    )


def scaling_example() -> None:
    width = 21
    words = ((0, 1), (2, 3), (4, 5), (6, 7), (8,), (9,), (10,), (11,))
    r_output, returned_work = inverse_scaling_outputs(words[2], tuple(range(12, 21)))
    outputs = (*words[:2], r_output, *words[3:])
    labels = (
        (2, 2, 1, 1, 0, 0, 0, 1),
        (1, 0, 2, 0, 1, 1, 0, 1),
        (3, 3, 0, 2, 0, 1, 1, 0),
    )
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    amplitudes = torch.tensor((1, 1j, -1), dtype=torch.complex128) / sqrt(3)
    for row, amp in zip(labels, amplitudes, strict=True):
        u, value, r, s, branch, parity, marker, active = row
        branch ^= marker ^ parity
        if active:
            assert value % 2 == 0
            value //= 2
        r = 2 * r % 3
        if parity:
            u, value, r, s = value, u, s, r
        parity ^= int(s % 2 == 0)
        output = (u, value, r, s, branch, parity, marker, active)
        state[pack(row, words, width)] = amp
        expected[pack(output, outputs, width)] = amp
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    circuit.append(CoefficientScaling())
    result = run_dynamic(circuit, shots=1, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(-1), expected, rtol=0, atol=1e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    circuit.append(CoefficientScaling(adjoint=True))
    restored = run_dynamic(circuit, shots=1, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(-1), state, rtol=0, atol=1e-12
    )
    assert returned_work[0] == words[2][0]
    print(
        "Modulo-three scaling: complex amplitudes and original output roles correct; nine returned work bits zero."
    )
    print("Original adjoint restored the input without materializing a SWAP.")
    print("Measured scaling primitives:", dict(counts))
    print(
        "Independent source estimates: default CSwap 40 T; exact CSwap 55 T; five measurements; twenty-one qubits."
    )


def main() -> None:
    arithmetic_example()
    scaling_example()


if __name__ == "__main__":
    main()
