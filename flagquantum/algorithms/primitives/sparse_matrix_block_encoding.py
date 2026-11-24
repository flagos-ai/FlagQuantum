# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SparseMatrix.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/sparse_matrix.py

"""Compose the source sparse-access matrix block from its supplied oracles."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import cast

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
    bit_length,
    smax,
)
from ...compiler.registers.layout.auto_partition import AutoPartition
from ...compiler.registers.preparation.sparse_matrix_component import (
    sparse_matrix_signature,
)
from ...compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...compiler.registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_resource_formula,
)
from ...compiler.registers.primitives.swap import (
    source_swap_resource_formula,
    swap_register_ir,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import Instruction
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation, _validate_block_work
from .black_box_prepare import BlackBoxPrepare, _identity_prepare
from .sparse_matrix_oracles import EntryOracle, RowColumnOracle


def source_sparse_matrix_signature(width: SymbolicInt) -> Signature:
    """Declare the ordered sparse-access ports before concrete binding."""
    return sparse_matrix_signature(width)


def _source_sparse_diffusion_resource_formula(
    width: SymbolicInt, sparsity: SymbolicInt, *, legacy_shims: bool
) -> SymbolicResourceFormula:
    if sparsity == 1:
        raise ValueError("Source singleton diffusion lacks its target port")
    uniform = source_uniform_preparation_resource_formula(
        sparsity, legacy_shims=legacy_shims
    )
    active = bit_length(sparsity - 1)
    partition = AutoPartition(
        (("target", ((None, width - active), ("target", active))),)
    )
    return replace(
        uniform, logical_qubits=partition.logical_width(uniform.logical_qubits)
    )


def source_symbolic_sparse_matrix_resource_formula(
    width: SymbolicInt,
    sparsity: SymbolicInt,
    *,
    row: SymbolicResourceFormula,
    column: SymbolicResourceFormula,
    entry: SymbolicResourceFormula,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Compose the six locked calls with independently declared oracle costs.

    Supply the oracle formulas in their execution direction. Forward uses
    row adjoint, column forward and entry forward. The adjoint uses the
    opposite directions. The sparse block's symbolic peak is the maximum
    of its signature and the callee peaks. Diffusion alone includes the
    internal unused index prefix through AutoPartition.
    """
    if type(width) is int and type(sparsity) is int:
        raise ValueError("Use the concrete sparse block model for resolved dimensions")
    diffusion = _source_sparse_diffusion_resource_formula(
        width, sparsity, legacy_shims=legacy_shims
    )
    swap = source_swap_resource_formula(width)
    children = (diffusion, column, entry, swap, row, diffusion)
    return SymbolicResourceFormula(
        logical_qubits=cast(
            SymbolicInt,
            smax(
                source_sparse_matrix_signature(width).n_qubits(),
                *(child.logical_qubits for child in children),
            ),
        ),
        **{
            field.name: sum(getattr(child, field.name) for child in children)
            for field in fields(SymbolicResourceFormula)
            if field.name != "logical_qubits"
        },
    )


@dataclass(frozen=True)
class SparseMatrixBlockEncoding:
    """Bind a sparse-access block with an ancilla ordered as flag then index."""

    row_oracle: RowColumnOracle
    column_oracle: RowColumnOracle
    entry_oracle: EntryOracle
    eps: SymbolicReal = 0

    def __post_init__(self) -> None:
        n = self.system_bitsize
        if type(n) is not int or n < 1:
            raise ValueError("Sparse block binding requires a positive concrete width")
        if (
            n != self.row_oracle.system_bitsize
            or n != self.column_oracle.system_bitsize
        ):
            raise ValueError("Row, column and entry oracles must share the index width")
        if self.row_oracle.num_nonzero != self.column_oracle.num_nonzero:
            raise ValueError("Sparse block row and column sparsities must match")
        if type(self.alpha) is not int or not 1 <= self.alpha <= 1 << n:
            raise ValueError("Sparse block sparsity must fit its index width")

    def __str__(self) -> str:
        return "B[SparseMatrix]"

    @property
    def system_bitsize(self) -> int:
        return self.entry_oracle.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return self.system_bitsize + 1

    @property
    def resource_bitsize(self) -> int:
        return 0

    @property
    def signature(self) -> Signature:
        return source_sparse_matrix_signature(self.system_bitsize)

    @property
    def alpha(self) -> int:
        return self.row_oracle.num_nonzero

    @property
    def epsilon(self) -> SymbolicReal:
        return self.eps

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def requirements(self) -> tuple[int, int]:
        """Reuse the largest clean allocation across sequential oracle calls."""
        _, work, slots = uniform_preparation_requirements(self.alpha)
        children = (self.row_oracle, self.column_oracle, self.entry_oracle)
        demands = ((work, slots), *(child.requirements() for child in children))
        return max(w for w, _ in demands), max(b for _, b in demands)

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        """Keep the six source calls and the stage-local bystander widths."""
        if type(adjoint) is not bool:
            raise ValueError("Sparse block direction must be boolean")
        n = self.system_bitsize
        total = 2 * n + 1
        active, _, _ = uniform_preparation_requirements(self.alpha)
        diffusion = source_uniform_preparation_cost(self.alpha)
        first = CostCall(diffusion, bystander_qubits=total - active)
        last = CostCall(
            replace(diffusion, name="UniformPreparation adjoint"),
            bystander_qubits=total - active,
        )
        col = CostCall(
            self.column_oracle.resource_model(adjoint=adjoint), bystander_qubits=1
        )
        entry = CostCall(self.entry_oracle.resource_model(adjoint=adjoint))
        swap = CostCall(
            ComponentCost("Swap", 2 * n, GateCost(clifford=n)), bystander_qubits=1
        )
        row = CostCall(
            self.row_oracle.resource_model(adjoint=not adjoint), bystander_qubits=1
        )
        middle = (row, swap, entry, col) if adjoint else (col, entry, swap, row)
        return ComponentCost("SparseMatrix", total, calls=(first, *middle, last))

    def make_on(
        self,
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind preparation, column oracle, entry, swap, row adjoint and unprepare."""
        if len(system) != self.system_bitsize or len(ancilla) != self.ancilla_bitsize:
            raise ValueError(
                "Sparse block ports must match the ordered index and flag widths"
            )
        if (
            type(adjoint) is not bool
            or (len(workspace), len(classical_bits)) != self.requirements()
        ):
            raise ValueError(
                "Sparse block needs its declared work, slots and direction"
            )
        through = (*system, *ancilla)
        _validate_block_work(through, workspace, classical_bits)
        flag, ordinal = ancilla[0], ancilla[1:]
        width, prep_work, prep_slots = uniform_preparation_requirements(self.alpha)
        target = ordinal[len(ordinal) - width :]
        before = uniform_preparation_ir(
            self.alpha, target, workspace[:prep_work], classical_bits[:prep_slots]
        )

        def index_oracle(
            oracle: RowColumnOracle, inverse: bool
        ) -> tuple[Instruction, ...]:
            work, slots = oracle.requirements()
            return oracle.emit(
                ordinal,
                system,
                workspace[:work],
                classical_bits[:slots],
                adjoint=inverse,
            )

        def entry() -> tuple[Instruction, ...]:
            work, slots = self.entry_oracle.requirements()
            return self.entry_oracle.emit(
                flag,
                ordinal,
                system,
                workspace[:work],
                classical_bits[:slots],
                adjoint=adjoint,
            )

        swap = swap_register_ir(ordinal, system)
        middle = (
            (
                *index_oracle(self.row_oracle, False),
                *swap,
                *entry(),
                *index_oracle(self.column_oracle, True),
            )
            if adjoint
            else (
                *index_oracle(self.column_oracle, False),
                *entry(),
                *swap,
                *index_oracle(self.row_oracle, True),
            )
        )
        after = uniform_preparation_ir(
            self.alpha,
            target,
            workspace[:prep_work],
            classical_bits[:prep_slots],
            adjoint=True,
        )
        return _BoundBlockOperation(
            (*before, *middle, *after), max((*through, *workspace)) + 1
        )
