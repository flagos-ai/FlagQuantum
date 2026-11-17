"""Supply independent user oracles to the native Hermitian sparse block.

The column permutation is l XOR i XOR all-ones. The entry rotation depends
on the conjunction of the low row and column bits, giving a symmetric,
nonuniform matrix. Neither class inherits a library implementation.
"""

from collections import Counter
from dataclasses import dataclass
from math import isfinite

from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_classification import (
    classify_resource_gate,
)
from flagquantum.compiler._component_trace.resource_generalization import RotationCall
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit


@dataclass(frozen=True)
class XorShiftColumn:
    """Map an ordinal bijectively within every full-support column."""

    system_bitsize: int

    @property
    def num_nonzero(self) -> int:
        return 1 << self.system_bitsize

    def requirements(self) -> tuple[int, int]:
        return 0, 0

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        return ComponentCost(
            "UserColumnInverse" if adjoint else "UserColumn",
            2 * self.system_bitsize,
            GateCost(clifford=2 * self.system_bitsize),
        )

    def emit(
        self,
        ordinal: tuple[int, ...],
        index: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        gates = tuple(
            gate
            for source, target in zip(index, ordinal, strict=True)
            for gate in (
                Instruction("cx", (source, target)),
                Instruction("x", (target,)),
            )
        )
        return tuple(reversed(gates)) if adjoint else gates


@dataclass(frozen=True)
class ConjunctionSqrtEntry:
    """Rotate by base + delta * (low_row AND low_column), with clean work."""

    system_bitsize: int
    base: float = 0.37
    delta: float = 0.62
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(angle) for angle in (self.base, self.delta)):
            raise ValueError("Entry rotation angles must be finite")

    def requirements(self) -> tuple[int, int]:
        return 1, 1

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        # One compute/erase pair, two CNOTs and three classified Ry rotations.
        # The model And is distinct from its emitted four-T implementation.
        families = Counter(
            classify_resource_gate(RotationCall("ry", angle))
            for angle in (self.base, self.delta / 2, -self.delta / 2)
        )
        return ComponentCost(
            "UserEntryInverse" if adjoint else "UserEntry",
            2 * self.system_bitsize + 2,
            GateCost(
                and_bloq=1,
                clifford=3 + families["clifford"],
                t=families["t"],
                rotation=families["rotation"],
                measurement=1,
            ),
        )

    def emit(
        self,
        flag: int,
        row: tuple[int, ...],
        column: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        (predicate,) = work
        (slot,) = slots
        rotation: tuple[Instruction, ...] = (
            Instruction("ry", (flag,), {"theta": self.delta / 2}),
            Instruction("cx", (predicate, flag)),
            Instruction("ry", (flag,), {"theta": -self.delta / 2}),
            Instruction("cx", (predicate, flag)),
        )
        base = Instruction(
            "ry", (flag,), {"theta": -self.base if adjoint else self.base}
        )
        if adjoint:
            rotation = tuple(
                Instruction("ry", gate.wires, {"theta": -gate.params["theta"]})
                if gate.name == "ry"
                else gate
                for gate in reversed(rotation)
            )
        body = (
            *and_compute_ir(row[-1], column[-1], predicate),
            *rotation,
            *and_uncompute_ir(row[-1], column[-1], predicate, slot),
        )
        return (*body, base) if adjoint else (base, *body)


def custom_hermitian_block(width: int) -> HermitianSparseMatrixBlockEncoding:
    """Compose user implementations through the existing oracle interfaces."""
    return HermitianSparseMatrixBlockEncoding(
        XorShiftColumn(width), ConjunctionSqrtEntry(width)
    )


if __name__ == "__main__":
    block = custom_hermitian_block(1)
    op = block.make_on(
        system=(0,), ancilla=(1, 2, 3), workspace=(4,), classical_bits=(0,)
    )
    circuit = DynamicCircuit(5).append(op)
    assert sum(g.name == "measure" for g in circuit.to_ir().instructions) == 2
    print("Custom sparse block: one clean work wire, two measured erasures.")
