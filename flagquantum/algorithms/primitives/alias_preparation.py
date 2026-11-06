# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 state_preparation_alias_sampling.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py

"""Coherent dense and sparse alias preparation on bound registers."""

from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.controlled_swap import controlled_swap_ir
from ...compiler.registers.less_equal import less_equal_ir, less_equal_requirements
from ...compiler.registers.source_qrom import source_qrom_ir, source_qrom_workspace_size
from ...compiler.registers.uniform_preparation import (
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...core.ir import Instruction
from .alias_sampling import AliasSamplingTable


def _datasets(table: AliasSamplingTable) -> tuple[tuple[int, ...], ...]:
    if table.selection_size == 1:
        return (table.keep,)
    if table.indices is None:
        return table.alternates, table.keep
    return table.indices, table.alternates, table.keep


def alias_preparation_requirements(table: AliasSamplingTable) -> tuple[int, int]:
    """Return the reusable clean work and slots, excluding logical junk."""
    rows = len(table.keep)
    width, uniform_work, uniform_slots = uniform_preparation_requirements(rows)
    lookup = source_qrom_workspace_size(0, (width,), _datasets(table))
    compare_work, compare_slots = less_equal_requirements(table.mu, table.mu)
    return max(uniform_work, lookup, compare_work), max(
        uniform_slots, lookup, compare_slots
    )


def alias_preparation_ir(
    table: AliasSamplingTable,
    selection: tuple[int, ...],
    sigma: tuple[int, ...],
    alternate: tuple[int, ...],
    keep: tuple[int, ...],
    flag: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    sparse_index: tuple[int, ...] = (),
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Prepare or undo the source selection and logical junk registers.

    Words are big endian. Start selection, sigma, alternate, keep, flag and
    sparse index at zero to prepare a coefficient state. The inverse restores
    these on the matching state. Work always starts and ends at zero.
    """
    rows = len(table.keep)
    data_width = (table.selection_size - 1).bit_length()
    address_width, uniform_work, uniform_slots = uniform_preparation_requirements(rows)
    required, slots = alias_preparation_requirements(table)
    sparse = table.indices is not None
    if (
        len(selection) != data_width
        or len(alternate) != data_width
        or len(sigma) != table.mu
        or len(keep) != table.mu
        or len(sparse_index) != (address_width if sparse else 0)
    ):
        raise ValueError("Alias preparation registers must match the table layout")
    if (
        len(workspace) != required
        or len(classical_bits) != slots
        or len(set(classical_bits)) != slots
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("Alias preparation requires clean work and distinct slots")
    if type(adjoint) is not bool:
        raise ValueError("Alias preparation direction must be boolean")
    _validate_wires(
        (*selection, *sigma, *sparse_index, *alternate, *keep, flag, *workspace)
    )

    address = sparse_index if sparse else selection
    uniform = uniform_preparation_ir(
        rows,
        address,
        workspace[:uniform_work],
        classical_bits[:uniform_slots],
        adjoint=adjoint,
    )
    hadamards = tuple(Instruction("h", (q,)) for q in sigma)
    datasets = _datasets(table)
    words: tuple[tuple[int, ...], ...] = (keep,)
    if data_width:
        words = (selection, alternate, keep) if sparse else (alternate, keep)
    lookup_size = source_qrom_workspace_size(0, (address_width,), datasets)
    lookup = source_qrom_ir(
        (),
        (address,),
        (rows,),
        tuple((word,) for word in words),
        datasets,
        workspace[:lookup_size],
        classical_bits[:lookup_size],
    )
    cmp_work, cmp_slots = less_equal_requirements(table.mu, table.mu)
    compare = less_equal_ir(
        keep, sigma, flag, workspace[:cmp_work], classical_bits[:cmp_slots]
    )
    swap = controlled_swap_ir(flag, alternate, selection) if selection else ()
    if adjoint:
        return (*swap, *compare, *lookup, *hadamards, *uniform)
    return (*uniform, *hadamards, *lookup, *compare, *swap)
