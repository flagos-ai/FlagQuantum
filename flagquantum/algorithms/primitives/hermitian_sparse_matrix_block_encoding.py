# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 sparse_matrix_hermitian.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/sparse_matrix_hermitian.py

"""Conjugate the index and flag swaps by the source sparse preparation."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import cached_property
from math import sqrt
from typing import Protocol, cast

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ...compiler.registers.control.control_predicate import (
    ControlRegister,
    bind_control_registers,
)
from ...compiler.registers.control.controlled_component import controlled_component_ir
from ...compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
    controlled_component_resource_formula,
)
from ...compiler.registers.lookup.matrix_entry_component import entry_signature
from ...compiler.registers.preparation.hermitian_sparse_matrix_component import (
    hermitian_sparse_matrix_signature,
)
from ...compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...compiler.registers.primitives.controlled_swap import controlled_swap_ir
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
from .sparse_matrix_block_encoding import _source_sparse_diffusion_resource_formula
from .sparse_matrix_oracles import (
    EntryOracle,
    RowColumnOracle,
    UniformEntryOracle,
    source_top_left_oracle_resource_formula,
    source_uniform_entry_oracle_resource_formula,
)


def source_sqrt_entry_signature(width: SymbolicInt) -> Signature:
    """Declare the source flag and two ordered scalar index words."""
    return entry_signature(width)


def source_hermitian_sparse_signature(
    width: SymbolicInt, *, controlled: bool = False
) -> Signature:
    """Declare the full source block ports for concrete or symbolic widths."""
    return hermitian_sparse_matrix_signature(width, controlled=controlled)


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

    @property
    def signature(self) -> Signature:
        return source_sqrt_entry_signature(self.system_bitsize)

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
    def signature(self) -> Signature:
        return source_hermitian_sparse_signature(self.system_bitsize)

    def __str__(self) -> str:
        return "B[SparseMatrixHermitian]"

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

    def controlled_resource_model(
        self, control_registers: tuple[ControlRegister, ...]
    ) -> ComponentCost:
        """Count the concrete predicate around the existing positive block."""
        return controlled_component_cost(
            self.resource_model(controlled=True),
            control_registers,
            target_width=self.system_bitsize + self.ancilla_bitsize,
        )

    def make_controlled_on(
        self,
        *,
        controls: tuple[tuple[int, ...], ...],
        control_registers: tuple[ControlRegister, ...],
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        control_workspace: tuple[int, ...] = (),
        control_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind a negative or structured control with its own live scratch."""
        wires, _ = bind_control_registers(controls, control_registers)
        through = (*wires, *system, *ancilla)
        work = (*workspace, *control_workspace)
        _validate_block_work(through, work, (*classical_bits, *control_classical_bits))
        gates = controlled_component_ir(
            controls,
            control_registers,
            lambda predicate: (
                self.make_on(
                    system=system,
                    ancilla=ancilla,
                    control=predicate,
                    workspace=workspace,
                    classical_bits=classical_bits,
                    adjoint=adjoint,
                ).instructions
            ),
            control_workspace,
            control_classical_bits,
        )
        return _BoundBlockOperation(gates, max((*through, *work)) + 1)

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


def source_hermitian_sparse_fallback_formula(
    width: SymbolicInt,
    sparsity: SymbolicInt,
    *,
    column: tuple[SymbolicResourceFormula, SymbolicResourceFormula],
    entry: tuple[SymbolicResourceFormula, SymbolicResourceFormula],
    controlled: bool = False,
    control_registers: tuple[ControlRegister, ...] = (),
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Estimate the eight source callees when the composite is symbolic.

    Oracle pairs are forward and adjoint. Always keeps their formulas and
    those for diffusion unchanged under an added positive control. Only
    the exchanges become controlled. Structured controls wrap that
    positive child with the shared predicate model. Use the concrete
    component model for live bystanders in a resolved circuit.
    """
    if type(controlled) is not bool or (controlled and control_registers):
        raise ValueError("Hermitian sparse control requires one control specification")
    positive = controlled or bool(control_registers)
    diffusion = _source_sparse_diffusion_resource_formula(
        width, sparsity, legacy_shims=legacy_shims
    )

    def swap(bits: SymbolicInt) -> SymbolicResourceFormula:
        if positive:
            return SymbolicResourceFormula(0, 0, 0, 2 * bits + 1, cswap=bits)
        return source_swap_resource_formula(bits)

    children = (
        diffusion,
        column[0],
        entry[0],
        swap(width),
        swap(1),
        entry[1],
        column[1],
        diffusion,
    )
    target_width = 2 * width + 2
    model = SymbolicResourceFormula(
        logical_qubits=cast(
            SymbolicInt,
            smax(
                target_width + int(positive),
                *(child.logical_qubits for child in children),
            ),
        ),
        **{
            field.name: sum(getattr(child, field.name) for child in children)
            for field in fields(SymbolicResourceFormula)
            if field.name != "logical_qubits"
        },
    )
    if not control_registers:
        return model
    return controlled_component_resource_formula(
        ComponentCost("Positive Hermitian sparse block", 0),
        model,
        control_registers,
        target_width=target_width,
        legacy_shims=legacy_shims,
    )


def source_uniform_hermitian_sparse_resource_formula(
    width: SymbolicInt, *, controlled: bool = False, legacy_shims: bool = False
) -> SymbolicResourceFormula:
    """Model the full-support uniform instance through the source callees."""
    column = source_top_left_oracle_resource_formula(width)
    entry = source_uniform_entry_oracle_resource_formula(width)
    return source_hermitian_sparse_fallback_formula(
        width,
        2**width,
        column=(column, column),
        entry=(entry, entry),
        controlled=controlled,
        legacy_shims=legacy_shims,
    )


def source_uniform_hermitian_sparse_hadamards(width: SymbolicInt) -> SymbolicInt:
    """Count the direct Hadamard leaf in the locked full-support call graph."""
    # Concrete power-of-two preparation emits one H per bit in each direction.
    # The symbolic preparation declares three H per bit even at full support.
    return (2 if type(width) is int else 6) * width
