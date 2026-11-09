"""Run python -m examples.algorithms.classical_swap_values."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import torch

import flagquantum as fq
from flagquantum.compiler.registers.primitives.controlled_swap import controlled_swap_ir
from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
    relative_controlled_swap_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.classical_swap import controlled_swap_classical_values


@dataclass(frozen=True)
class Exchange:
    """Use the existing exact or relative-phase controlled exchange emitter."""

    relative: bool = False
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = (
            relative_controlled_swap_ir(0, (1,), (2,), adjoint=self.inverse)
            if self.relative
            else controlled_swap_ir(0, (1,), (2,))
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    permutation = torch.zeros_like(initial)
    for control, left, right, spectator in product((0, 1), repeat=4):
        _, x, y = controlled_swap_classical_values(control, left, right)
        before = (control << 3) | (left << 2) | (right << 1) | spectator
        after = (control << 3) | (x << 2) | (y << 1) | spectator
        permutation[after] = initial[before]
    for relative in (False, True):
        circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
        circuit.append(Exchange(relative=relative))
        actual = circuit.state().flatten()
        if relative:
            torch.testing.assert_close(
                actual.abs(), permutation.abs(), rtol=0, atol=2e-12
            )
            assert not torch.allclose(actual, permutation, rtol=0, atol=2e-12)
        else:
            torch.testing.assert_close(actual, permutation, rtol=0, atol=2e-12)
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        assert counts["t"] + counts["tdg"] == (4 if relative else 7)
        assert counts["measure"] == 0
        circuit.append(Exchange(relative=relative, inverse=True))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, rtol=0, atol=2e-12
        )
    print(
        "Exact exchange: 7 emitted T, source CSwap weight 4 T equivalents (7 when selected)"
    )
    print("Relative exchange: 4 emitted T, source model 4 T equivalents")
    print(
        "Both: 0 work, 0 measurements, full complex inverse restores input and spectator"
    )
    print(
        "The relative exchange retains quantum phases absent from its classical callback"
    )


if __name__ == "__main__":
    main()
