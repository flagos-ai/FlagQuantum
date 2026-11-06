"""Compose the source-faithful windowed Montgomery circuit and its adjoint.

Run: python -m examples.algorithms.register_windowed_mod_mul
"""

from collections import Counter
from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.registers.windowed_mod_mul import windowed_mod_mul_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit


@dataclass(frozen=True)
class WindowedMontgomeryMultiplication:
    """Bind n=3, w=1, p=3 to the original input and output wire layout."""

    adjoint: bool = False

    @property
    def target_wires(self) -> tuple[int, ...]:
        """Logical accumulator order after this direction."""
        return (6, 7, 8) if self.adjoint else (7, 8, 13)

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates, target = windowed_mod_mul_ir(
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            12,
            3,
            (13,),
            ((14, 15, 16, 17),),
            (18, 19, 20),
            tuple(range(6)),
            adjoint=self.adjoint,
        )
        assert target == self.target_wires
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    circuit = DynamicCircuit(21)
    op = WindowedMontgomeryMultiplication()
    circuit.append(op)
    print("Target, retained indices, reduction flag:", op.target_wires, (9, 10, 11), 12)
    print(
        "Forward primitive counts:",
        dict(Counter(g.name for g in circuit.to_ir().instructions)),
    )
    inverse = WindowedMontgomeryMultiplication(adjoint=True)
    circuit.append(inverse)
    print("After uncomputation:", inverse.target_wires)


if __name__ == "__main__":
    main()
