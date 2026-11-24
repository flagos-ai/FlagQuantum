# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call order follows Qualtran v0.7.0 SelectSwapQROM and its Adjoint.

"""Compose blocked XOR lookup while restoring owned or borrowed banks."""

from math import prod

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .blocked_lookup_component import (
    BankBindings,
    load_lookup_banks,
    partition_lookup_selection,
    route_lookup_banks,
    unpartition_lookup_selection,
)
from .qrom_table import QromTable
from .source_qroam_clean import QroamCleanTable


def select_swap_qrom_program(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    *,
    selection_widths: tuple[int, ...] | None = None,
    num_controls: int = 0,
    dirty: bool = True,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain logical THRU words and restore the complete temporary bank state.

    Bank words and bits are MSB-first. Forward dirty banks are borrowed. Source
    inverse allocation comes from the original default Free, hence is clean.
    QROM is an XOR involution using its forward measurement/cleanup schedule;
    adjoint reverses calls and selects exact routing inverses, not measurements.
    """
    if type(dirty) is not bool or type(adjoint) is not bool:
        raise ValueError("Select-swap bank and direction flags must be boolean")
    if len(target_widths) != len(table.datasets) or any(
        type(n) is not int or n < 0 for n in target_widths
    ):
        raise ValueError("Select-swap requires one nonnegative width per dataset")
    widths = (
        selection_widths
        if selection_widths is not None
        else tuple((n - 1).bit_length() for n in table.shape)
    )
    layout = QromTable(
        table.shape,
        widths,
        target_widths,
        ((),) * len(target_widths),
        num_controls,
        table.datasets,
    )
    layout._require_concrete_dimensions()
    table._high_widths(widths)
    signature = layout.signature
    builder = ComponentProgramBuilder(signature)
    banks: BankBindings = {}
    targets = {}
    for i, width in enumerate(target_widths):
        if not width:
            continue
        dtype = signature.get_left(f"target{i}_").dtype
        banks[i] = np.asarray(
            [
                builder.allocate(dtype=dtype, dirty=dirty and not adjoint)
                for _ in range(prod(table.block_shape))
            ],
            dtype=object,
        ).reshape(table.block_shape)
        targets[i] = builder.inputs[f"target{i}_"]
    parts, high, low = partition_lookup_selection(builder, signature, table, widths)
    control = builder.inputs.get("control")
    stages: tuple[str, ...] = ("load", "route", "copy", "unroute", "load")
    if dirty:
        stages += ("route", "copy", "unroute")
    if adjoint:
        stages = tuple(
            {"route": "unroute", "unroute": "route"}.get(stage, stage)
            for stage in reversed(stages)
        )
    order = tuple(reversed(banks)) if adjoint else tuple(banks)
    for stage in stages:
        if stage == "load":
            control = load_lookup_banks(
                builder,
                table,
                target_widths,
                widths,
                num_controls,
                control,
                high,
                banks,
            )
        elif stage in ("route", "unroute"):
            ordered = {i: banks[i] for i in order}
            route_lookup_banks(
                builder, table, target_widths, low, ordered, adjoint=stage == "unroute"
            )
            banks.update(ordered)
        else:
            for i in order:
                bank = banks[i]
                word = bank.flat[0]
                assert isinstance(word, RegisterBinding)
                result = builder.call_program(
                    bitwise_word_program(word.register.dtype, "xor"),
                    {"x": word, "y": targets[i]},
                )
                bank.flat[0], targets[i] = result["x"], result["y"]
    for bank in banks.values():
        for word in bank.flat:
            assert isinstance(word, RegisterBinding)
            builder.free(word, dirty=dirty and adjoint)
    outputs = unpartition_lookup_selection(builder, signature, table, parts, high, low)
    if num_controls:
        outputs["control"] = control
    outputs.update({f"target{i}_": target for i, target in targets.items()})
    return builder.finalize(outputs)
