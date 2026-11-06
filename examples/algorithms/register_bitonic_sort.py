"""Append original bitonic merge/sort and restore words with their exact adjoints.

Run: python -m examples.algorithms.register_bitonic_sort
"""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.bitonic_merge import bitonic_merge_ir
from flagquantum.compiler.registers.bitonic_sort import bitonic_sort_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class BitonicWords:
    """Bind four one-bit words, original histories and two shared clean work bits."""

    operation: Literal["merge", "sort"]
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        count = 4 if self.operation == "merge" else 6
        emit = bitonic_merge_ir if self.operation == "merge" else bitonic_sort_ir
        gates = emit(
            tuple((i,) for i in range(4)),
            tuple(range(4, 4 + count)),
            (4 + count, 5 + count),
            (0,),
            adjoint=self.inverse,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    operations: tuple[Literal["merge", "sort"], ...] = ("merge", "sort")
    for operation in operations:
        count = 4 if operation == "merge" else 6
        width = 6 + count
        labels = [
            v
            for v in product((0, 1), repeat=4)
            if operation == "sort" or (v[0] <= v[1] and v[2] <= v[3])
        ]
        v = torch.arange(len(labels), dtype=torch.float64)
        amps = torch.complex(v + 1, 3 * v - 5)
        amps /= amps.norm()
        state = torch.zeros(1 << width, dtype=torch.complex128)
        expected = torch.zeros_like(state)
        # Independent classical histories for this four-word teaching network.
        pairs: tuple[tuple[int, int], ...] = ((0, 3), (1, 2), (0, 1), (2, 3))
        if operation == "sort":
            pairs = ((0, 1), (2, 3), *pairs)
        for values, amp in zip(labels, amps, strict=True):
            words = list(values)
            history = []
            for i, j in pairs:
                history.append(int(words[i] > words[j]))
                words[i], words[j] = min(words[i], words[j]), max(words[i], words[j])
            assert tuple(words) == tuple(sorted(values))
            before = after = 0
            for bit in values:
                before = (before << 1) | bit
            for bit in (*words, *history):
                after = (after << 1) | bit
            state[before << (count + 2)] = amp
            expected[after << 2] = amp
        for inverse in (False, True):
            initial, target = (expected, state) if inverse else (state, expected)
            circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
            circuit.append(BitonicWords(operation, inverse))
            result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(8, -1),
                target.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            circuit.append(BitonicWords(operation, not inverse))
            restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                initial.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                operation,
                "inverse",
                inverse,
                ": complex amplitudes correct; two work bits zero; original inverse restores words/history",
            )
            print("Measured primitives:", dict(counts))
        print(
            "Logical histories:",
            count,
            "; independent exact/default T estimates:",
            11 * count,
            8 * count,
            "; measurements:",
            count,
            "; total qubits:",
            width,
        )


if __name__ == "__main__":
    main()
