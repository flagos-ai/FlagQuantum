# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call construction follows Qualtran v0.7.0 QROAMClean.build_composite_bloq.

"""Compose clean lookup banks from shared partition, QROM and routing owners."""

from math import prod

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import bounded_unsigned_register_type
from ....core.ir.program.registers import Register
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..layout.component_partition import partition_component_program
from .qroam_clean_metadata import qroam_clean_signature
from .qrom_component import qrom_program
from .qrom_table import QromTable
from .source_qroam_clean import QroamCleanTable
from .swap_with_zero_component import swap_with_zero_program


def qroam_clean_program(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    *,
    selection_widths: tuple[int, ...] | None = None,
    num_controls: int = 0,
) -> ProgramRecord:
    """Return selected targets and retained banks with caller-owned RIGHT ports.

    Dataset and bank cells are row-major; bits are MSB-first. Only QROM predicate
    work is private and measurement-erased. This forward component does not
    supply a body for the source's model-only adjoint leaf.
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
    high_widths = table._high_widths(widths)
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
    selected = [reg for reg in signature if reg.name.startswith("selection")]
    axes = [i for i, extent in enumerate(table.shape) if extent > 1]
    blocked = any(table.log_block_sizes)
    parts: dict[int, tuple[Register, ...]] = {}
    high: dict[int, BindingCollection] = {}
    low: dict[int, BindingCollection] = {}
    for axis, register in zip(axes, selected, strict=True):
        if blocked:
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
    batched = QromTable(
        table.batch_shape,
        high_widths,
        target_widths,
        (table.block_shape,) * len(target_widths),
        num_controls,
        table.batched_data,
    )
    qrom = qrom_program(batched, reserve_workspace=True)
    qrom_inputs: dict[str, object] = {
        f"target{i}_": values for i, values in banks.items()
    }
    if num_controls:
        qrom_inputs["control"] = builder.inputs["control"]
    high_axes = [i for i, extent in enumerate(table.batch_shape) if extent > 1]
    for axis, register in zip(
        high_axes,
        (reg for reg in batched.signature if reg.name.startswith("selection")),
        strict=True,
    ):
        qrom_inputs[register.name] = high[axis]
    loaded = builder.call_program(qrom, qrom_inputs)
    outputs: dict[str, object] = {}
    if num_controls:
        outputs["control"] = loaded["control"]
    for axis, register in zip(
        high_axes,
        (reg for reg in batched.signature if reg.name.startswith("selection")),
        strict=True,
    ):
        high[axis] = loaded[register.name]
    for i, width in enumerate(target_widths):
        if not width:
            continue
        bank = np.asarray(loaded[f"target{i}_"])
        if blocked:
            route = swap_with_zero_program(
                table.log_block_sizes, width, table.block_shape
            )
            route_ports = route.functions[0].registers
            assert route_ports is not None
            route_inputs = {
                reg.name: low[axis]
                for axis, reg in zip(axes, route_ports[:-1], strict=True)
            }
            routed = builder.call_program(route, {**route_inputs, "targets": bank})
            for axis, reg in zip(axes, route_ports[:-1], strict=True):
                low[axis] = routed[reg.name]
            bank = np.asarray(routed["targets"])
        outputs[f"target{i}_"] = bank.flat[0]
        if bank.size > 1:
            outputs[f"junk_target{i}_"] = bank.reshape(-1)[1:]
    for axis, register in zip(axes, selected, strict=True):
        if blocked:
            name = f"unpartition_{axis}"
            joined = builder.call_program(
                partition_component_program(parts[axis], name, True),
                {"l": high[axis], "k": low[axis]},
                entry=name,
            )
            outputs[register.name] = joined["x"]
        else:
            outputs[register.name] = high[axis]
    return builder.finalize(outputs)
