"""Compose the source-faithful windowed Montgomery circuit and its adjoint.

Run: python -m examples.algorithms.register_windowed_mod_mul
"""

from collections import Counter
from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.registers.modular.single_window_mod_mul import (
    single_window_qroam_table,
)
from flagquantum.compiler.registers.modular.windowed_mod_mul import windowed_mod_mul_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit


@dataclass(frozen=True)
class WindowedMontgomeryMultiplication:
    """Bind the original Montgomery inputs, retained history, and clean work."""

    adjoint: bool = False
    width: int = 3
    window: int = 1
    modulus: int = 3

    @property
    def target_wires(self) -> tuple[int, ...]:
        """Logical accumulator order after this direction."""
        n, w = self.width, self.window
        if self.adjoint:
            return tuple(range(2 * n, 3 * n))
        return (*range(2 * n + w, 3 * n), *range(4 * n + 1, 4 * n + 1 + w))

    @property
    def history_wires(self) -> tuple[int, ...]:
        return tuple(range(3 * self.width, 4 * self.width))

    @property
    def reduction_wire(self) -> int:
        return 4 * self.width

    @property
    def qubit_count(self) -> int:
        n, w = self.width, self.window
        bank = single_window_qroam_table(n, w, self.modulus).block_shape[0]
        return 4 * n + 1 + w + bank * (n + w) + n + w - 1

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        n, w = self.width, self.window
        bank_start = self.reduction_wire + 1 + w
        work_start = self.qubit_count - (n + w - 1)
        bank = tuple(
            tuple(range(start, start + n + w))
            for start in range(bank_start, work_start, n + w)
        )
        gates, target = windowed_mod_mul_ir(
            tuple(range(n)),
            tuple(range(n, 2 * n)),
            tuple(range(2 * n, 3 * n)),
            self.history_wires,
            self.reduction_wire,
            self.modulus,
            tuple(range(self.reduction_wire + 1, bank_start)),
            bank,
            tuple(range(work_start, self.qubit_count)),
            tuple(range(2 * n)),
            adjoint=self.adjoint,
        )
        assert target == self.target_wires
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    op = WindowedMontgomeryMultiplication()
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    print(
        "Target, retained indices, reduction flag:",
        op.target_wires,
        op.history_wires,
        op.reduction_wire,
    )
    print(
        "Forward primitive counts:",
        dict(Counter(g.name for g in circuit.to_ir().instructions)),
    )
    inverse = WindowedMontgomeryMultiplication(adjoint=True)
    circuit.append(inverse)
    print("After uncomputation:", inverse.target_wires)


if __name__ == "__main__":
    main()
