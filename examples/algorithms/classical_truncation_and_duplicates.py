"""Run python -m examples.algorithms.classical_truncation_and_duplicates."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement, product

import numpy as np
import torch

import flagquantum as fq
from examples.algorithms.register_sorted_duplicates import SortedDuplicates
from flagquantum.compiler.registers.bits.sign_extension import sign_extend_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_signed_truncation import (
    signed_truncate_classical_value,
)
from flagquantum.simulation.classical_sorted_duplicates import (
    sorted_duplicates_classical_values,
)


@dataclass(frozen=True)
class TruncateSignedWord:
    """Clear caller-owned redundant prefix bits, retaining the narrower word."""

    retained: tuple[int, ...]
    prefix: tuple[int, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(
            CircuitIR(circuit.n_qubits, sign_extend_ir(self.retained, self.prefix))
        )


def main() -> None:
    for ones in (False, True):
        assert signed_truncate_classical_value(-1, 4, 2, ones_complement=ones) == -1
    grid = torch.arange(1, 9, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, spectator in product(range(4), (0, 1)):
        prefix = 3 if word & 2 else 0
        initial[(((prefix << 2) | word) << 1) | spectator] = weights[
            2 * word + spectator
        ]
        expected[(word << 1) | spectator] = weights[2 * word + spectator]
    op = TruncateSignedWord((2, 3), (0, 1))
    circuit = fq.Circuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts == {"cx": 3}
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print(
        "Truncation emitted: 3 CX, 0 T, 0 measurements; 2 redundant input bits cleared"
    )
    print(
        "Pinned source QEC model: 1 logical Clifford component (separate counting basis)"
    )

    cases = list(product(combinations_with_replacement((0, 1), 2), (0, 1)))
    grid = torch.arange(1, len(cases) + 1, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for index, (values, flag) in enumerate(cases):
        output = sorted_duplicates_classical_values(
            {"xs": np.array(values), "flag": flag}
        )
        data = (values[0] << 2) | (values[1] << 1) | flag
        updated = (data & ~1) | int(output["flag"])
        initial[data << 5] = weights[index]
        expected[updated << 5] = weights[index]
    dynamic = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    dynamic.append(SortedDuplicates())
    result = run_dynamic(dynamic, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in dynamic.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 4 and counts["measure"] == 1
    print(
        "Duplicates emitted: 4 T, 1 measurement, 5 reserved clean work, 8 declared wires"
    )
    print("Pinned source model: 4 T equivalents, 1 measurement, 7 peak logical qubits")
    print(
        "All complex amplitudes agree; duplicate work zero; sign word/spectator restored"
    )


if __name__ == "__main__":
    main()
