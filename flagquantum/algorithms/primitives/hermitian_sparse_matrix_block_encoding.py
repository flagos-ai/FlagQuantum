# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 sparse_matrix_hermitian.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/sparse_matrix_hermitian.py

"""Conjugate the index and flag swaps by the source sparse preparation."""

from dataclasses import dataclass, replace
from functools import cached_property
from math import sqrt
from typing import Protocol

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal
from ...compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...compiler.registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_resource_formula,
)
from ...compiler.registers.primitives.controlled_swap import controlled_swap_ir
from ...compiler.registers.primitives.swap import swap_register_ir
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import Instruction
from ._block_binding import _BoundBlockOperation, _validate_block_work
from .black_box_prepare import BlackBoxPrepare, _identity_prepare
from .sparse_matrix_oracles import EntryOracle, RowColumnOracle, UniformEntryOracle


class SqrtEntryOracle(EntryOracle, Protocol):
    """Supply the square-root amplitude and its declared precision."""

    @property
    def epsilon(self) -> SymbolicReal: ...


@dataclass(frozen=True)
class UniformSqrtEntryOracle:
    """Use the existing uniform entry rotation with a square-root amplitude."""

    system_bitsize: int
    entry: float
    epsilon: SymbolicReal = 1e-11

    def __post_init__(self) -> None:
        if not 0 <= self.entry <= 1:
            raise ValueError("Square-root entry must be in [0, 1]")
        self._rotation

    @cached_property
    def _rotation(self) -> UniformEntryOracle:
        return UniformEntryOracle(self.system_bitsize, sqrt(self.entry))

    def requirements(self) -> tuple[int, int]:
        return self._rotation.requirements()

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        name = "UniformSqrtEntry adjoint" if adjoint else "UniformSqrtEntry"
        return replace(self._rotation.resource_model(adjoint=adjoint), name=name)

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
        return self._rotation.emit(flag, row, column, work, slots, adjoint=adjoint)


@dataclass(frozen=True)
class HermitianSparseMatrixBlockEncoding:
    """Bind a self-adjoint block with ancilla packed as flag, ordinal, flag."""

    column_oracle: RowColumnOracle
    entry_oracle: SqrtEntryOracle
    eps: SymbolicReal = 0

    def __post_init__(self) -> None:
        n = self.system_bitsize
        if type(n) is not int or n < 1 or self.column_oracle.system_bitsize != n:
            raise ValueError(
                "Hermitian sparse oracles must share a positive concrete width"
            )
        if type(self.alpha) is not int or not 1 <= self.alpha <= 1 << n:
            raise ValueError("Hermitian sparse sparsity must fit its index width")

    @property
    def system_bitsize(self) -> int:
        return self.entry_oracle.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return self.system_bitsize + 2

    @property
    def resource_bitsize(self) -> int:
        return 0

    @property
    def alpha(self) -> int:
        return self.column_oracle.num_nonzero

    @property
    def epsilon(self) -> SymbolicReal:
        return self.eps

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def requirements(self) -> tuple[int, int]:
        _, work, slots = uniform_preparation_requirements(self.alpha)
        stages = (
            (work, slots),
            self.column_oracle.requirements(),
            self.entry_oracle.requirements(),
        )
        return max(w for w, _ in stages), max(b for _, b in stages)

    def resource_model(self, *, controlled: bool = False) -> ComponentCost:
        """Count the source Always calls and the two optionally controlled swaps."""
        if type(controlled) is not bool:
            raise ValueError("Hermitian sparse control flag must be boolean")
        n = self.system_bitsize
        controls = int(controlled)
        total = 2 * n + 2 + controls
        active, _, _ = uniform_preparation_requirements(self.alpha)
        diffusion = source_uniform_preparation_cost(self.alpha)
        first = CostCall(diffusion, bystander_qubits=total - active)
        last = CostCall(
            replace(diffusion, name="UniformPreparation adjoint"),
            bystander_qubits=total - active,
        )
        column = CostCall(
            self.column_oracle.resource_model(), bystander_qubits=2 + controls
        )
        inverse_column = CostCall(
            self.column_oracle.resource_model(adjoint=True),
            bystander_qubits=2 + controls,
        )
        entry = CostCall(
            self.entry_oracle.resource_model(), bystander_qubits=1 + controls
        )
        inverse_entry = CostCall(
            self.entry_oracle.resource_model(adjoint=True),
            bystander_qubits=1 + controls,
        )

        def swap(width: int) -> CostCall:
            gates = GateCost(cswap=width) if controlled else GateCost(clifford=width)
            node = ComponentCost(
                "CSwap" if controlled else "Swap", 2 * width + controls, gates
            )
            return CostCall(node, bystander_qubits=total - node.qubits)

        return ComponentCost(
            "HermitianSparseMatrix",
            total,
            calls=(
                first,
                column,
                entry,
                swap(n),
                swap(1),
                inverse_entry,
                inverse_column,
                last,
            ),
        )

    def make_on(
        self,
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        control: int | None = None,
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Append the same conjugation in either direction; control only the two swaps."""
        if len(system) != self.system_bitsize or len(ancilla) != self.ancilla_bitsize:
            raise ValueError("Hermitian sparse ports must match the declared widths")
        if (
            type(adjoint) is not bool
            or (len(workspace), len(classical_bits)) != self.requirements()
        ):
            raise ValueError(
                "Hermitian sparse block needs its declared work, slots and direction"
            )
        through = (*system, *ancilla, *((control,) if control is not None else ()))
        _validate_block_work(through, workspace, classical_bits)
        first_flag, ordinal, second_flag = ancilla[0], ancilla[1:-1], ancilla[-1]
        width, prep_work, prep_slots = uniform_preparation_requirements(self.alpha)
        target = ordinal[len(ordinal) - width :]
        column_work, column_slots = self.column_oracle.requirements()
        entry_work, entry_slots = self.entry_oracle.requirements()

        def pair(
            quantum: int, classical: int
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return workspace[:quantum], classical_bits[:classical]

        before = uniform_preparation_ir(
            self.alpha, target, *pair(prep_work, prep_slots)
        )
        column = self.column_oracle.emit(
            ordinal, system, *pair(column_work, column_slots)
        )
        entry = self.entry_oracle.emit(
            second_flag, ordinal, system, *pair(entry_work, entry_slots)
        )
        exchange = (
            (
                *swap_register_ir(ordinal, system),
                *swap_register_ir((first_flag,), (second_flag,)),
            )
            if control is None
            else (
                *controlled_swap_ir(control, ordinal, system),
                *controlled_swap_ir(control, (first_flag,), (second_flag,)),
            )
        )
        inverse_entry = self.entry_oracle.emit(
            second_flag, ordinal, system, *pair(entry_work, entry_slots), adjoint=True
        )
        inverse_column = self.column_oracle.emit(
            ordinal, system, *pair(column_work, column_slots), adjoint=True
        )
        after = uniform_preparation_ir(
            self.alpha, target, *pair(prep_work, prep_slots), adjoint=True
        )
        return _BoundBlockOperation(
            (
                *before,
                *column,
                *entry,
                *exchange,
                *inverse_entry,
                *inverse_column,
                *after,
            ),
            max((*through, *workspace)) + 1,
        )


def source_uniform_hermitian_sparse_resource_formula(
    width: SymbolicInt, *, controlled: bool = False
) -> SymbolicResourceFormula:
    """Model the original full-support uniform example without physical binding."""
    if type(controlled) is not bool:
        raise ValueError("Hermitian sparse control flag must be boolean")
    diffusion = source_uniform_preparation_resource_formula(2**width)
    swaps = width + 1
    return SymbolicResourceFormula(
        and_bloq=2 * diffusion.and_bloq,
        clifford=2 * diffusion.clifford + (0 if controlled else swaps),
        measurement=2 * diffusion.measurement,
        logical_qubits=2 * width + 2 + int(controlled),
        cswap=swaps if controlled else 0,
        rotation=2 * diffusion.rotation + 2,
    )


def source_uniform_hermitian_sparse_hadamards(width: SymbolicInt) -> SymbolicInt:
    """Count the direct Hadamard leaf in the locked full-support call graph."""
    # Concrete power-of-two preparation emits one H per bit in each direction.
    # The symbolic preparation declares three H per bit even at full support.
    return (2 if type(width) is int else 6) * width
