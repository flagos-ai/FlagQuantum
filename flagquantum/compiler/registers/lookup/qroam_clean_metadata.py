# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROAMClean and QROAMCleanAdjointWrapper.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qroam_clean.py

"""Source QROAM signatures and padded adjoint table literals."""

from __future__ import annotations

from itertools import product
from math import prod

from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..._component_trace.symbolic_math import SymbolicInt, bit_length
from .source_qroam_clean import QroamCleanTable
from .swap_with_zero import swap_with_zero_stages


def _index_ports(
    shape: tuple[SymbolicInt, ...],
    widths: tuple[SymbolicInt, ...] | None,
    controls: SymbolicInt,
) -> tuple[Register, ...]:
    widths = (
        tuple(bit_length(extent - 1) for extent in shape) if widths is None else widths
    )
    selections = tuple(
        bounded_unsigned_register_type(width, extent)
        for extent, width in zip(shape, widths, strict=True)
        if extent != 1
    )
    indices = (
        (Register("selection", selections[0]),)
        if len(selections) == 1
        else tuple(Register(f"selection{i}", word) for i, word in enumerate(selections))
    )
    return (
        (Register("control", opaque_register_type(controls)),) if controls != 0 else ()
    ) + indices


def qroam_clean_signature(
    shape: tuple[SymbolicInt, ...],
    widths: tuple[SymbolicInt, ...],
    block_logs: tuple[SymbolicInt, ...],
    *,
    selection_widths: tuple[SymbolicInt, ...] | None = None,
    controls: SymbolicInt = 0,
) -> Signature:
    """Declare scalar targets followed by all dataset-major retained words."""
    if not shape or len(shape) != len(block_logs):
        raise ValueError("QROAM signature requires matching nonempty axes")
    targets = tuple(
        Register(f"target{i}_", opaque_register_type(width), side=RegisterSide.RIGHT)
        for i, width in enumerate(widths)
        if width != 0
    )
    cells = prod(2**log for log in block_logs)
    retained = tuple(
        Register(
            f"junk_{target.name}",
            target.dtype,
            shape=(cells - 1,),
            side=RegisterSide.RIGHT,
        )
        for target in targets
        if cells != 1
    )
    return Signature(
        (*_index_ports(shape, selection_widths, controls), *targets, *retained)
    )


def qroam_clean_adjoint_leaf_signature(
    shape: tuple[SymbolicInt, ...],
    widths: tuple[SymbolicInt, ...],
    block_logs: tuple[SymbolicInt, ...],
    *,
    controls: SymbolicInt = 0,
) -> Signature:
    """Declare the consumed bank-shaped ports; shape is the leaf's actual table extent."""
    if not shape or len(shape) != len(block_logs):
        raise ValueError("QROAM adjoint signature requires matching nonempty axes")
    bank_shape = tuple(2**log for log in block_logs)
    return Signature(
        (
            *_index_ports(shape, None, controls),
            *(
                Register(
                    f"target{i}_",
                    opaque_register_type(width),
                    shape=bank_shape,
                    side=RegisterSide.LEFT,
                )
                for i, width in enumerate(widths)
                if width != 0
            ),
        )
    )


def permuted_qroam_erase_data(
    table: QroamCleanTable,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return padded address rows, each containing the source unsigned bank words.

    Rows use interleaved high/low indices in the original axis order.
    They describe the opaque source erasure leaf's data, without encoding
    the forward relative phase or making a gate-execution claim for that leaf.
    """
    blocks = table.block_shape
    batches = table.batch_shape
    cells = prod(blocks)
    padded = tuple(
        extent * block for extent, block in zip(batches, blocks, strict=True)
    )
    swaps = tuple(swap_with_zero_stages(table.log_block_sizes, blocks))
    result: list[list[tuple[int, ...]]] = [[] for _ in table.datasets]
    for address in product(*(range(extent) for extent in padded)):
        high, low = zip(
            *(
                divmod(value, block)
                for value, block in zip(address, blocks, strict=True)
            ),
            strict=True,
        )
        batch = 0
        for value, extent in zip(high, batches, strict=True):
            batch = batch * extent + value
        order = list(range(cells))
        for axis, bit, left, right in swaps:
            if low[axis] >> bit & 1:
                order[left], order[right] = order[right], order[left]
        for rows, data in zip(result, table.batched_data, strict=True):
            rows.append(tuple(data[batch * cells + cell] for cell in order))
    return tuple(tuple(rows) for rows in result)
