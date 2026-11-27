# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectSwapQROM.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/select_swap_qrom.py

"""Direct calls of a blocked XOR lookup before QEC cost aggregation."""

from collections import Counter

from ..._component_trace.symbolic_math import SymbolicInt
from .qrom_declaration import QromDeclaration
from .qrom_table import QromTable
from .select_swap_callees import SelectSwapRouteCall as SelectSwapRouteCall
from .select_swap_callees import SelectSwapXorCall as SelectSwapXorCall
from .select_swap_declaration import SelectSwapDeclaration

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
    base = QromDeclaration(
        tuple(table.shape + target for target in table.target_shapes),
        (
            (None,) * len(table.target_shapes)
            if table.datasets is None
            else table.datasets
        ),
        table.selection_bitsizes,
        table.target_bitsizes,
        table.target_shapes,
        table.num_controls,
    )
    declaration = SelectSwapDeclaration(base, log_block_sizes, dirty)
    calls: Counter[SelectSwapCall] = Counter()
    for child, count in declaration.calls().items():
        calls[
            child.materialize() if isinstance(child, QromDeclaration) else child
        ] += count
    return dict(calls)
