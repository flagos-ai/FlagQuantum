# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Reuses locked Qualtran v0.7.0 batch XOR and relative-phase bank routing.

"""Extend clean lookup to caller-owned arbitrary banks with an exact inverse."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .blocked_lookup_component import (
    BankBindings,
    load_lookup_banks,
    partition_lookup_selection,
    route_lookup_banks,
    unpartition_lookup_selection,
)
from .qroam_clean_metadata import qroam_clean_signature
from .source_qroam_clean import QroamCleanTable


def qroam_bank_unitary_pair(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    *,
    selection_widths: tuple[int, ...] | None = None,
    num_controls: int = 0,
) -> tuple[ProgramRecord, ProgramRecord]:
    """XOR into arbitrary banks then route, or invert routing before the XOR.

    All bank cells are caller-owned THRU words, in the clean lookup's selected
    then dataset-major retained order. This constructs a different physical
    program, without bank allocation or measured-bank erasure. Only the QROM
    predicate scratch is privately allocated and measurement-erased. On zero
    bank inputs its forward action is the original clean lookup isometry.
    """
    widths = (
        selection_widths
        if selection_widths is not None
        else tuple((n - 1).bit_length() for n in table.shape)
    )
    if len(target_widths) != len(table.datasets) or any(
        type(width) is not int or width < 0 for width in target_widths
    ):
        raise ValueError("Bank unitary requires one nonnegative width per dataset")
    if type(num_controls) is not int or num_controls < 0:
        raise ValueError("Bank unitary requires a nonnegative control count")
    table._high_widths(widths)
    original = qroam_clean_signature(
        table.shape,
        target_widths,
        table.log_block_sizes,
        selection_widths=widths,
        controls=num_controls,
    )
    signature = Signature(
        tuple(Register(reg.name, reg.dtype, shape=reg.shape) for reg in original)
    )

    def build(inverse: bool) -> ProgramRecord:
        builder = ComponentProgramBuilder(signature)
        banks: BankBindings = {}
        for i, width in enumerate(target_widths):
            if width:
                cells = (
                    builder.inputs[f"target{i}_"],
                    *np.asarray(builder.inputs.get(f"junk_target{i}_", ())).flat,
                )
                banks[i] = np.asarray(cells, dtype=object).reshape(table.block_shape)
        parts, high, low = partition_lookup_selection(builder, signature, table, widths)
        if inverse:
            reversed_banks = dict(reversed(tuple(banks.items())))
            route_lookup_banks(
                builder, table, target_widths, low, reversed_banks, adjoint=True
            )
            banks.update(reversed_banks)
        control = load_lookup_banks(
            builder,
            table,
            target_widths,
            widths,
            num_controls,
            builder.inputs.get("control"),
            high,
            banks,
        )
        if not inverse:
            route_lookup_banks(builder, table, target_widths, low, banks)
        outputs = unpartition_lookup_selection(
            builder, signature, table, parts, high, low
        )
        if num_controls:
            outputs["control"] = control
        for i, bank in banks.items():
            outputs[f"target{i}_"] = bank.flat[0]
            if bank.size > 1:
                outputs[f"junk_target{i}_"] = bank.reshape(-1)[1:]
        return builder.finalize(outputs)

    return build(False), build(True)
