"""Copy a half-comparison flag and erase its retained difference.

Run: python -m examples.algorithms.register_half_comparison
Expected: every two-bit input/result amplitude correct, six temporary wires zero
for signed and unsigned inputs and all four relations. No reference packages.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.bits.bitwise import xor_ir
from flagquantum.compiler.registers.comparison.half_ordering import (
    half_compare_ir,
    half_compare_outputs,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class HalfComparison:
    """Bind two-bit inputs, logical difference/flag, two sign bits and one result."""

    relation: Literal["gt", "ge", "lt", "le"]
    signed: bool

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        args = ((0, 1), (2, 3), (5, 6, 7), 4, (8, 9))
        compute = half_compare_ir(*args, relation=self.relation, signed=self.signed)
        erase = half_compare_ir(
            *args,
            relation=self.relation,
            signed=self.signed,
            adjoint=True,
            classical_bits=(0, 1),
        )
        gates = (*compute, *xor_ir((4,), (10,)), *erase)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    data = torch.arange(16, dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 11, dtype=torch.complex128)
    state[data << 7] = amps
    for signed in (False, True):
        left, right = data >> 2, data & 3
        if signed:
            left, right = left - ((left >> 1) << 2), right - ((right >> 1) << 2)
        predicates = {
            "gt": left > right,
            "ge": left >= right,
            "lt": left < right,
            "le": left <= right,
        }
        relations: tuple[Literal["gt", "ge", "lt", "le"], ...] = (
            "gt",
            "ge",
            "lt",
            "le",
        )
        for relation in relations:
            expected = torch.zeros_like(state)
            expected[(data << 7) | predicates[relation].to(torch.int64)] = amps
            circuit = DynamicCircuit(11, inputs=state, dtype=torch.complex128).append(
                HalfComparison(relation, signed)
            )
            actual = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                actual.final_states.reshape(32, -1),
                expected.expand(32, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            print(
                f"signed={signed}, relation={relation}: all input/result complex amplitudes correct, six temporary wires zero"
            )
            print("Measured gates including result copy:", dict(counts))
            print(
                "Logical adjoint output ports:",
                half_compare_outputs(
                    (0, 1), (2, 3), (5, 6, 7), 4, relation=relation, adjoint=True
                ),
            )
    print(
        "Independent source model per compute/erase: 8 T, 2 measurements; result copy is one CNOT."
    )


if __name__ == "__main__":
    main()
