# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 state_preparation_alias_sampling.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py

"""Coherent dense and sparse alias preparation on bound registers."""

from collections.abc import Mapping

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.controlled_swap import controlled_swap_ir
from ...compiler.registers.less_equal import (
    less_equal_ir,
    less_equal_requirements,
    source_equal_width_less_equal_cost,
)
from ...compiler.registers.qrom_table import QromTable
from ...compiler.registers.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...core.ir import Instruction
from .alias_sampling import AliasSamplingTable
from .black_box_prepare import BlackBoxPrepare


def _lookup_table(table: AliasSamplingTable) -> QromTable:
    rows = len(table.keep)
    width = (table.selection_size - 1).bit_length()
    datasets: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]
    if table.selection_size == 1:
        datasets, targets = (table.keep,), (table.mu,)
    elif table.indices is None:
        datasets, targets = (table.alternates, table.keep), (width, table.mu)
    else:
        datasets = (table.indices, table.alternates, table.keep)
        targets = (width, width, table.mu)
    return QromTable(
        (rows,),
        ((rows - 1).bit_length(),),
        targets,
        ((),) * len(targets),
        datasets=datasets,
    )


def alias_preparation_requirements(table: AliasSamplingTable) -> tuple[int, int]:
    """Return the reusable clean work and slots, excluding logical junk."""
    rows = len(table.keep)
    _, uniform_work, uniform_slots = uniform_preparation_requirements(rows)
    lookup = _lookup_table(table).workspace_size
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
    words: tuple[tuple[int, ...], ...] = (keep,)
    if data_width:
        words = (selection, alternate, keep) if sparse else (alternate, keep)
    lookup_table = _lookup_table(table)
    lookup_size = lookup_table.workspace_size
    lookup = lookup_table.bind_packed(
        (*address, *(wire for word in words for wire in word)),
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


def alias_black_box_prepare(table: AliasSamplingTable) -> BlackBoxPrepare:
    """Expose source-ordered selection and packed logical junk for alias PREPARE."""
    width = (table.selection_size - 1).bit_length()
    address_width = (len(table.keep) - 1).bit_length()
    sparse = (("sparse_index", address_width),) if table.indices is not None else ()
    junk = (
        ("sigma_mu", table.mu),
        *sparse,
        ("alt", width),
        ("keep", table.mu),
        ("less_than_equal", 1),
    )

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return alias_preparation_ir(
            table,
            ports["selection"],
            ports["sigma_mu"],
            ports["alt"],
            ports["keep"],
            ports["less_than_equal"][0],
            work,
            slots,
            sparse_index=ports.get("sparse_index", ()),
            adjoint=adjoint,
        )

    return BlackBoxPrepare((("selection", width),), junk, emit, table.normalization)


def source_dense_alias_preparation_cost(
    table: AliasSamplingTable, *, legacy_shims: bool = False
) -> ComponentCost:
    """Count the dense source call graph and active preparation width."""
    if table.indices is not None or table.selection_size < 2:
        raise ValueError("Dense alias costs require at least two rows")
    return _source_alias_preparation_cost(table, legacy_shims=legacy_shims)


def source_sparse_alias_preparation_cost(
    table: AliasSamplingTable, *, legacy_shims: bool = False
) -> ComponentCost:
    """Count the source three-table preparation on at least two support rows."""
    if table.indices is None or len(table.indices) < 2:
        raise ValueError("Sparse alias costs require at least two support rows")
    return _source_alias_preparation_cost(table, legacy_shims=legacy_shims)


def _source_alias_preparation_cost(
    table: AliasSamplingTable, *, legacy_shims: bool
) -> ComponentCost:
    rows, mu = len(table.keep), table.mu
    selection = (table.selection_size - 1).bit_length()
    address = (rows - 1).bit_length()
    sparse = table.indices is not None
    sparse_width = address if sparse else 0
    width = 2 * selection + sparse_width + 2 * mu + 1
    compare = source_equal_width_less_equal_cost(mu, legacy_shims=legacy_shims)
    lookup = _lookup_table(table).resource_model(legacy_shims=legacy_shims)
    return ComponentCost(
        (
            "SparseStatePreparationAliasSampling"
            if sparse
            else "StatePreparationAliasSampling"
        ),
        width,
        GateCost(clifford=mu, cswap=selection),
        (
            CostCall(
                source_uniform_preparation_cost(rows, legacy_shims=legacy_shims),
                bystander_qubits=width - address,
            ),
            CostCall(lookup, bystander_qubits=mu + 1),
            CostCall(compare, bystander_qubits=2 * selection + sparse_width),
        ),
    )
