# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call construction follows Qualtran v0.7.0 QROAMClean.build_composite_bloq.

"""Compose clean lookup banks from shared partition, QROM and routing owners."""

from math import prod

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program import ProgramRecord
from ..component_builder import ComponentProgramBuilder
from .blocked_lookup_component import (
    load_lookup_banks,
    partition_lookup_selection,
    route_lookup_banks,
    unpartition_lookup_selection,
)
from .qroam_clean_metadata import qroam_clean_signature
from .source_qroam_clean import QroamCleanTable


def qroam_clean_program(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    *,
    selection_widths: tuple[int, ...] | None = None,
    num_controls: int = 0,
    reserve_workspace: bool = True,
) -> ProgramRecord:
    """Return selected targets and retained banks with caller-owned RIGHT ports.

    Dataset and bank cells are row-major; bits are MSB-first. Only QROM predicate
    work is private and measurement-erased. This forward component does not
    supply a body for the source's model-only adjoint leaf.
    Source call-graph binding can disable the native lexical workspace frame;
    that frame adds bookkeeping calls without changing the lookup operation.
    """
    widths = (
        selection_widths
        if selection_widths is not None
        else tuple((extent - 1).bit_length() for extent in table.shape)
    )
    if len(target_widths) != len(table.datasets) or any(
        type(n) is not int or n < 0 for n in target_widths
    ):
        raise ValueError(
            "Clean lookup requires one nonnegative concrete width per dataset"
        )
    if type(num_controls) is not int or num_controls < 0:
        raise ValueError("Clean lookup requires a nonnegative concrete control count")
    table._high_widths(widths)
    signature = qroam_clean_signature(
        table.shape,
        target_widths,
        table.log_block_sizes,
        selection_widths=widths,
        controls=num_controls,
    )
    builder = ComponentProgramBuilder(signature)
    banks: dict[int, NDArray[np.object_]] = {}
    for i, width in enumerate(target_widths):
        if not width:
            continue
        dtype = signature.get_right(f"target{i}_").dtype
        cells = [
            builder.allocate(dtype=dtype) for _ in range(prod(table.block_shape) - 1)
        ]
        cells.insert(0, builder.allocate(dtype=dtype))
        banks[i] = np.asarray(cells, dtype=object).reshape(table.block_shape)
    parts, high, low = partition_lookup_selection(builder, signature, table, widths)
    control = load_lookup_banks(
        builder,
        table,
        target_widths,
        widths,
        num_controls,
        builder.inputs.get("control"),
        high,
        banks,
        reserve_workspace=reserve_workspace,
    )
    route_lookup_banks(builder, table, target_widths, low, banks)
    outputs = unpartition_lookup_selection(builder, signature, table, parts, high, low)
    if num_controls:
        outputs["control"] = control
    for i, bank in banks.items():
        outputs[f"target{i}_"] = bank.flat[0]
        if bank.size > 1:
            outputs[f"junk_target{i}_"] = bank.reshape(-1)[1:]
    return builder.finalize(outputs)
