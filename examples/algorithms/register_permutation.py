"""Use ordered basis-state cycles with explicit clean work and measured cleanup.

Run: python -m examples.algorithms.register_permutation
Expected: all three-bit amplitudes and an entangled spectator preserved under
an ordered permutation, three work wires zero, inverse restoring the input.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.permutation import permutation_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class BasisPermutation:
    """Bind three target bits, marker and two carries; wire six is a spectator."""

    cycles: tuple[tuple[int, ...], ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = permutation_ir((0, 1, 2), self.cycles, 3, (4, 5), (0, 1), size=8)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    cycles = ((0, 2, 5), (2, 7))  # Overlap makes the composition order observable.
    data = torch.arange(16, dtype=torch.int64)
    x, spectator = data >> 1, data & 1
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(128, dtype=torch.complex128)
    state[(x << 4) | spectator] = amps
    mapped = x.clone()
    for cycle in cycles:
        before = mapped.clone()
        for i, value in enumerate(cycle):
            mapped[before == value] = cycle[(i + 1) % len(cycle)]
    expected = torch.zeros_like(state)
    expected[(mapped << 4) | spectator] = amps
    circuit = DynamicCircuit(7, inputs=state, dtype=torch.complex128).append(
        BasisPermutation(cycles)
    )
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )
    inverse = tuple(c[::-1] for c in reversed(cycles))
    restored = run_dynamic(
        circuit.append(BasisPermutation(inverse)),
        shots=32,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
    forward = BasisPermutation(cycles).append_to(DynamicCircuit(7))
    print(
        "All complex amplitudes, spectator and three clean work wires verified; inverse restores input."
    )
    print(
        "Measured forward gates:",
        dict(Counter(g.name for g in forward.to_ir().instructions)),
    )
    print(
        "Independent source model: 56 T, 14 measurements; three quantum work wires, two reused classical slots."
    )


if __name__ == "__main__":
    main()
