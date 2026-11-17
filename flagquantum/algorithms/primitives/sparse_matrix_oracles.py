# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 sparse_matrix.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/sparse_matrix.py

"""Sparse matrix index and entry oracles with caller-owned work."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from math import acos, pi
from typing import Protocol, cast

import numpy as np

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, smax
from ...compiler.registers.arithmetic.addition import (
    add_adjoint_ir,
    add_constant_adjoint_ir,
    add_constant_ir,
    add_ir,
)
from ...compiler.registers.arithmetic.arithmetic_resource_model import source_add_cost
from ...compiler.registers.arithmetic.constant_add_resource_model import (
    source_constant_add_cost,
    source_constant_add_resource_formula,
)
from ...compiler.registers.lookup.qrom_table import QromTable
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
)
from ...core.ir import Instruction
from ._block_binding import _validate_block_work


class RowColumnOracle(Protocol):
    """Map an ordinal to its row or column index reversibly."""

    @property
    def system_bitsize(self) -> int: ...
    @property
    def num_nonzero(self) -> int: ...

    def requirements(self) -> tuple[int, int]: ...
    def resource_model(self, *, adjoint: bool = False) -> ComponentCost: ...
    def emit(
        self,
        ordinal: tuple[int, ...],
        index: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]: ...


class EntryOracle(Protocol):
    """Rotate the entry flag while retaining the row and column indices."""

    @property
    def system_bitsize(self) -> int: ...

    def requirements(self) -> tuple[int, int]: ...
    def resource_model(self, *, adjoint: bool = False) -> ComponentCost: ...
    def emit(
        self,
        flag: int,
        row: tuple[int, ...],
        column: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]: ...


def source_top_left_oracle_resource_formula(
    width: SymbolicInt,
) -> SymbolicResourceFormula:
    """Count the identity index map, retaining its two live words."""
    return SymbolicResourceFormula(0, 0, 0, 2 * width)


def source_uniform_entry_oracle_resource_formula(
    width: SymbolicInt,
) -> SymbolicResourceFormula:
    """Count the single source Ry on the flag with both indices live."""
    return SymbolicResourceFormula(0, 0, 0, 2 * width + 1, rotation=1)


def source_symbolic_banded_oracle_resource_formula(
    width: SymbolicInt,
    bandsize: SymbolicInt,
    *,
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Price source Add and signed AddK with the symbolic peak fallback."""
    if type(width) is int and type(bandsize) is int:
        raise ValueError("Use the concrete banded oracle model for resolved dimensions")
    add = source_add_resource_formula(width, width, legacy_shims=legacy_shims)
    shift = source_constant_add_resource_formula(
        width,
        -bandsize,
        encoding="twos_complement",
        adjoint=adjoint,
        legacy_shims=legacy_shims,
    )
    return SymbolicResourceFormula(
        logical_qubits=cast(
            SymbolicInt, smax(2 * width, add.logical_qubits, shift.logical_qubits)
        ),
        **{
            field.name: getattr(add, field.name) + getattr(shift, field.name)
            for field in fields(SymbolicResourceFormula)
            if field.name != "logical_qubits"
        },
    )


def _check_binding(
    through: tuple[int, ...],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    direction: bool,
    expected: tuple[int, int],
) -> None:
    if type(direction) is not bool or (len(work), len(slots)) != expected:
        raise ValueError("Sparse oracle needs its declared work, slots and direction")
    _validate_block_work(through, work, slots)


@dataclass(frozen=True)
class TopLeftRowColumnOracle:
    """Use the ordinal itself as the row or column position."""

    system_bitsize: int
    sparsity: int | None = None

    def __post_init__(self) -> None:
        if type(self.system_bitsize) is not int or self.system_bitsize < 1:
            raise ValueError("Top-left oracle width must be positive")
        if (
            type(self.num_nonzero) is not int
            or not 1 <= self.num_nonzero <= 1 << self.system_bitsize
        ):
            raise ValueError("Top-left oracle sparsity must fit its index width")

    @property
    def num_nonzero(self) -> int:
        return (1 << self.system_bitsize) if self.sparsity is None else self.sparsity

    def requirements(self) -> tuple[int, int]:
        return 0, 0

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        return ComponentCost("TopLeftRowColumn", 2 * self.system_bitsize)

    def emit(
        self,
        ordinal: tuple[int, ...],
        index: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        if len(ordinal) != len(index) or len(index) != self.system_bitsize:
            raise ValueError("Top-left oracle words must match its index width")
        _check_binding((*ordinal, *index), work, slots, adjoint, self.requirements())
        return ()


@dataclass(frozen=True)
class SymmetricBandedRowColumnOracle:
    """Map the ordinal to index + ordinal - bandsize modulo the index range."""

    system_bitsize: int
    bandsize: int

    def __post_init__(self) -> None:
        if type(self.system_bitsize) is not int or self.system_bitsize < 1:
            raise ValueError("Banded oracle width must be positive")
        if (
            type(self.bandsize) is not int
            or not 0 <= 2 * self.bandsize <= 1 << self.system_bitsize
        ):
            raise ValueError("Banded oracle bandsize must fit its index width")

    @property
    def num_nonzero(self) -> int:
        return 2 * self.bandsize + 1

    def requirements(self) -> tuple[int, int]:
        return 2 * self.system_bitsize - 1, self.system_bitsize - 1

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        n = self.system_bitsize
        add = CostCall(source_add_cost(n, n))
        shift = CostCall(
            source_constant_add_cost(
                n, -self.bandsize, encoding="twos_complement", adjoint=adjoint
            ),
            bystander_qubits=n,
        )
        return ComponentCost(
            "SymmetricBandedRowColumn",
            2 * n,
            calls=(shift, add) if adjoint else (add, shift),
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
        n = self.system_bitsize
        if len(ordinal) != n or len(index) != n:
            raise ValueError("Banded oracle words must match its index width")
        _check_binding((*ordinal, *index), work, slots, adjoint, self.requirements())
        carries, constant = work[: n - 1], work[n - 1 :]
        add = (add_adjoint_ir if adjoint else add_ir)(index, ordinal, carries, slots)
        shift = (add_constant_adjoint_ir if adjoint else add_constant_ir)(
            ordinal, -self.bandsize, constant, carries, slots
        )
        return (*shift, *add) if adjoint else (*add, *shift)


@dataclass(frozen=True)
class UniformEntryOracle:
    """Use the same real matrix entry for every row and column."""

    system_bitsize: int
    entry: float

    def __post_init__(self) -> None:
        if (
            type(self.system_bitsize) is not int
            or self.system_bitsize < 1
            or not -1 <= self.entry <= 1
        ):
            raise ValueError(
                "Uniform entry needs a positive width and entry in [-1, 1]"
            )

    def requirements(self) -> tuple[int, int]:
        return 0, 0

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        return ComponentCost(
            "UniformEntry", 1 + 2 * self.system_bitsize, GateCost(rotation=1)
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
        if len(row) != len(column) or len(row) != self.system_bitsize:
            raise ValueError("Uniform entry words must match its index width")
        _check_binding((flag, *row, *column), work, slots, adjoint, self.requirements())
        theta = 2 * acos(self.entry)
        return (Instruction("ry", (flag,), {"theta": -theta if adjoint else theta}),)


@dataclass(frozen=True)
class ExplicitEntryOracle:
    """Load a fixed-point arccos word, rotate the flag and unload the word."""

    system_bitsize: int
    data: tuple[tuple[float, ...], ...]
    entry_bitsize: int

    def __post_init__(self) -> None:
        if type(self.system_bitsize) is not int or self.system_bitsize < 1:
            raise ValueError("Explicit entry width must be positive")
        size = 1 << self.system_bitsize
        data = tuple(tuple(row) for row in self.data)
        if len(data) != size or any(len(row) != size for row in data):
            raise ValueError(
                "Explicit entries must form a square matching the index width"
            )
        if type(self.entry_bitsize) is not int or self.entry_bitsize < 1:
            raise ValueError("Explicit entry precision must be positive")
        if any(not 0 <= entry <= 1 for row in data for entry in row):
            raise ValueError("Explicit entries must be in [0, 1]")
        object.__setattr__(self, "data", data)

    @cached_property
    def table(self) -> QromTable:
        data = np.arccos(self.data) / np.pi * 2**self.entry_bitsize
        return QromTable.from_data(data, target_bitsizes=(self.entry_bitsize,))

    def requirements(self) -> tuple[int, int]:
        scratch = self.table.workspace_size
        return self.entry_bitsize + scratch, scratch

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        width = 2 * self.system_bitsize + 1
        lookup = CostCall(self.table.resource_model(), 2, bystander_qubits=1)
        rotations = CostCall(
            ComponentCost("ControlledRy", 2, GateCost(rotation=1)),
            self.entry_bitsize,
            bystander_qubits=width + self.entry_bitsize - 2,
        )
        return ComponentCost("ExplicitEntry", width, calls=(lookup, rotations))

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
        if len(row) != len(column) or len(row) != self.system_bitsize:
            raise ValueError("Explicit entry words must match its index width")
        _check_binding((flag, *row, *column), work, slots, adjoint, self.requirements())
        word, scratch = work[: self.entry_bitsize], work[self.entry_bitsize :]
        load = self.table.bind((), (row, column), ((word,),), scratch, slots)
        sign = -1 if adjoint else 1
        rotations = tuple(
            Instruction(
                "cry", (word[k], flag), {"theta": sign * 2 * pi * 2 ** -(k + 1)}
            )
            for k in (reversed(range(len(word))) if adjoint else range(len(word)))
        )
        return (*load, *rotations, *load)
