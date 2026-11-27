# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Shared call stages follow Qualtran v0.7.0 SelectSwapQROM and QROAMClean.

"""Compose shared selection views, blocked QROM calls and bank routing."""

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program.register_types import bounded_unsigned_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..layout.component_partition import partition_component_program
from .qrom_component import qrom_program
from .qrom_table import QromTable
from .select_swap_declaration import SelectSwapDeclaration
from .source_qroam_clean import QroamCleanTable
from .swap_with_zero_component import swap_with_zero_program

BankBindings = dict[int, NDArray[np.object_]]
SelectionBindings = dict[int, BindingCollection]
PartitionRegisters = dict[int, tuple[Register, ...]]


def partition_lookup_selection(
    builder: ComponentProgramBuilder,
    signature: Signature,
    table: QroamCleanTable | SelectSwapDeclaration,
    widths: tuple[int, ...],
) -> tuple[PartitionRegisters, SelectionBindings, SelectionBindings]:
    high_widths = table._high_widths(widths)
    selected = [reg for reg in signature if reg.name.startswith("selection")]
    axes = [i for i, extent in enumerate(table.shape) if extent > 1]
    parts: PartitionRegisters = {}
    high: SelectionBindings = {}
    low: SelectionBindings = {}
    for axis, register in zip(axes, selected, strict=True):
        if any(table.log_block_sizes):
            part = (
                Register(
                    "l",
                    bounded_unsigned_register_type(
                        high_widths[axis], 1 << high_widths[axis]
                    ),
                ),
                Register(
                    "k",
                    bounded_unsigned_register_type(
                        table.log_block_sizes[axis], table.block_shape[axis]
                    ),
                ),
            )
            parts[axis] = part
            name = f"partition_{axis}"
            values = builder.call_program(
                partition_component_program(part, name, False),
                {"x": builder.inputs[register.name]},
                entry=name,
            )
            high[axis], low[axis] = values["l"], values["k"]
        else:
            high[axis] = builder.inputs[register.name]
    return parts, high, low


def load_lookup_banks(
    builder: ComponentProgramBuilder,
    table: QroamCleanTable | SelectSwapDeclaration,
    target_widths: tuple[int, ...],
    selection_widths: tuple[int, ...],
    num_controls: int,
    control: BindingCollection | None,
    high: SelectionBindings,
    banks: BankBindings,
    *,
    reserve_workspace: bool = True,
) -> BindingCollection | None:
    batched = (
        table.qrom
        if isinstance(table, SelectSwapDeclaration)
        else QromTable(
            table.batch_shape,
            table._high_widths(selection_widths),
            target_widths,
            (table.block_shape,) * len(target_widths),
            num_controls,
            table.batched_data,
        )
    )
    inputs: dict[str, object] = {f"target{i}_": values for i, values in banks.items()}
    if num_controls:
        inputs["control"] = control
    axes = [i for i, extent in enumerate(table.batch_shape) if extent > 1]
    selected = [reg for reg in batched.signature if reg.name.startswith("selection")]
    for axis, register in zip(axes, selected, strict=True):
        inputs[register.name] = high[axis]
    loaded = builder.call_program(
        qrom_program(batched, reserve_workspace=reserve_workspace), inputs
    )
    for axis, register in zip(axes, selected, strict=True):
        high[axis] = loaded[register.name]
    for i in banks:
        banks[i] = np.asarray(loaded[f"target{i}_"])
    return loaded.get("control")


def route_lookup_banks(
    builder: ComponentProgramBuilder,
    table: QroamCleanTable | SelectSwapDeclaration,
    target_widths: tuple[int, ...],
    low: SelectionBindings,
    banks: BankBindings,
    *,
    adjoint: bool = False,
) -> None:
    if not any(table.log_block_sizes):
        return
    axes = [i for i, extent in enumerate(table.shape) if extent > 1]
    for i, bank in banks.items():
        route = swap_with_zero_program(
            table.log_block_sizes, target_widths[i], table.block_shape, adjoint=adjoint
        )
        ports = route.functions[0].registers
        assert ports is not None
        inputs = {
            reg.name: low[axis] for axis, reg in zip(axes, ports[:-1], strict=True)
        }
        routed = builder.call_program(route, {**inputs, "targets": bank})
        for axis, reg in zip(axes, ports[:-1], strict=True):
            low[axis] = routed[reg.name]
        banks[i] = np.asarray(routed["targets"])


def unpartition_lookup_selection(
    builder: ComponentProgramBuilder,
    signature: Signature,
    table: QroamCleanTable | SelectSwapDeclaration,
    parts: PartitionRegisters,
    high: SelectionBindings,
    low: SelectionBindings,
) -> dict[str, object]:
    selected = [reg for reg in signature if reg.name.startswith("selection")]
    axes = [i for i, extent in enumerate(table.shape) if extent > 1]
    outputs: dict[str, object] = {}
    for axis, register in zip(axes, selected, strict=True):
        if any(table.log_block_sizes):
            name = f"unpartition_{axis}"
            joined = builder.call_program(
                partition_component_program(parts[axis], name, True),
                {"l": high[axis], "k": low[axis]},
                entry=name,
            )
            outputs[register.name] = joined["x"]
        else:
            outputs[register.name] = high[axis]
    return outputs
