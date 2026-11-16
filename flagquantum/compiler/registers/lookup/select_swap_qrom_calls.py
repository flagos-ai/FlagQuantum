# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectSwapQROM.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/select_swap_qrom.py

"""Direct calls of a blocked XOR lookup before QEC cost aggregation."""

from collections import Counter
from dataclasses import dataclass, replace
from typing import cast

from ..._component_trace.symbolic_math import SymbolicInt
from .qrom_table import QromTable
from .source_qroam_clean import pack_blocked_datasets
from .symbolic_qroam_resource_model import _forward_logs, blocked_qrom_layout


@dataclass(frozen=True)
class SelectSwapRouteCall:
    """The original route parameters and the direction of the exchange."""

    width: SymbolicInt
    selectors: tuple[SymbolicInt, ...]
    banks: tuple[SymbolicInt, ...]
    adjoint: bool


@dataclass(frozen=True)
class SelectSwapXorCall:
    """XOR the routed word into a matching destination word."""

    width: SymbolicInt


SelectSwapCall = QromTable | SelectSwapRouteCall | SelectSwapXorCall


def select_swap_qrom_calls(
    table: QromTable,
    log_block_sizes: tuple[SymbolicInt, ...] | None = None,
    *,
    dirty: bool = True,
) -> dict[SelectSwapCall, int]:
    """Declare QROM, XOR and both route directions for either bank policy."""
    if type(dirty) is not bool:
        raise ValueError("Select-swap dirty mode must be boolean")
    if any(table.target_shapes):
        raise ValueError("Select-swap requires scalar targets")
    logs = (
        _forward_logs(table.shape, table.target_bitsizes, dirty)
        if log_block_sizes is None
        else log_block_sizes
    )
    blocks, high, batches = blocked_qrom_layout(
        table.shape, logs, table.selection_bitsizes
    )
    qrom = QromTable.from_shape(
        batches,
        table.target_bitsizes,
        target_shapes=(blocks,) * len(table.target_bitsizes),
        selection_bitsizes=high,
        num_controls=table.num_controls,
    )
    if table.datasets is not None and all(
        type(n) is int
        for n in (
            *table.shape,
            *logs,
            *table.target_bitsizes,
            *table.selection_bitsizes,
            table.num_controls,
        )
    ):
        data = pack_blocked_datasets(
            cast(tuple[int, ...], table.shape),
            cast(tuple[int, ...], blocks),
            cast(tuple[int, ...], batches),
            table.datasets,
        )
        qrom = replace(qrom, datasets=data)

    copies = 2 if dirty else 1
    calls: Counter[SelectSwapCall] = Counter({qrom: 2})
    has_routes = any(not isinstance(log, int) or log > 0 for log in logs)
    for width in table.target_bitsizes:
        if not isinstance(width, int) or width:
            calls[SelectSwapXorCall(width)] += copies
        if has_routes:
            calls[SelectSwapRouteCall(width, logs, blocks, False)] += copies
            calls[SelectSwapRouteCall(width, logs, blocks, True)] += copies
    return dict(calls)
