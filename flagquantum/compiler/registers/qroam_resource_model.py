# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROAMClean, SelectSwapQROM and SwapWithZero.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs

"""Compose analytical lookup costs from source QROM and bank routing."""

from dataclasses import dataclass
from math import prod

from .._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .qrom_resource_model import source_qrom_cost
from .source_qroam_clean import QroamCleanTable


def _batched_lookup_cost(
    table: QroamCleanTable,
    control_count: int,
    selection_widths: tuple[int, ...],
    target_widths: tuple[int, ...],
) -> ComponentCost:
    return source_qrom_cost(
        control_count,
        table._high_widths(selection_widths),
        table.batch_shape,
        tuple((prod(table.block_shape), width) for width in target_widths),
        table.batched_data,
    )


def _bank_route_costs(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    repetitions: int,
    logical_qubits: int,
) -> tuple[CostCall, ...]:
    banks = prod(table.block_shape)
    if banks == 1:
        return ()
    low = sum(table.log_block_sizes)
    calls = []
    for width in target_widths:
        active = low + banks * width
        calls.append(
            CostCall(
                ComponentCost(
                    "SwapWithZero",
                    active,
                    GateCost(
                        t=4 * width * (banks - 1),
                        clifford=(22 * width - 1) * (banks - 1),
                    ),
                ),
                repetitions,
                logical_qubits - active,
            )
        )
    return tuple(calls)


def source_qroam_clean_cost(
    table: QroamCleanTable,
    control_count: int,
    selection_widths: tuple[int, ...],
    target_widths: tuple[int, ...],
) -> ComponentCost:
    """Estimate the source forward lookup with all output banks retained."""
    qrom = _batched_lookup_cost(table, control_count, selection_widths, target_widths)
    low = sum(table.log_block_sizes)
    peak = qrom.qubits + low
    live = (
        control_count
        + sum(selection_widths)
        + prod(table.block_shape) * sum(target_widths)
    )
    routes = _bank_route_costs(table, target_widths, 1, live)
    return ComponentCost(
        "QROAMClean", peak, calls=(CostCall(qrom, bystander_qubits=low), *routes)
    )


def source_select_swap_qrom_cost(
    table: QroamCleanTable,
    control_count: int,
    selection_widths: tuple[int, ...],
    target_widths: tuple[int, ...],
    *,
    dirty: bool = True,
) -> ComponentCost:
    """Estimate the source XOR lookup with clean or borrowed output banks."""
    if type(dirty) is not bool:
        raise ValueError("Select-swap dirty mode must be boolean")
    qrom = _batched_lookup_cost(table, control_count, selection_widths, target_widths)
    toggles = 1 + dirty
    target = sum(target_widths)
    bystanders = sum(table.log_block_sizes) + target
    peak = qrom.qubits + bystanders
    live = (
        control_count + sum(selection_widths) + (1 + prod(table.block_shape)) * target
    )
    routes = _bank_route_costs(table, target_widths, 2 * toggles, live)
    return ComponentCost(
        "SelectSwapQROM",
        peak,
        GateCost(clifford=target * toggles),
        (CostCall(qrom, 2, bystanders), *routes),
    )


@dataclass(frozen=True)
class QroamErasureCosts:
    """The source's original-table wrapper and padded leaf cost declarations."""

    wrapper: ComponentCost
    padded_leaf: ComponentCost


def source_qroam_measurement_erase_cost(
    table: QroamCleanTable,
    control_count: int,
    target_widths: tuple[int, ...],
    *,
    erase_logs: tuple[int, ...] | None = None,
) -> QroamErasureCosts:
    """Model source measurement erasure, keeping wrapper and padded leaf separate.

    This model uses canonical selection widths and the forward output banks.
    The source erasure leaf has no executable gate decomposition.
    """
    selection_widths = tuple((extent - 1).bit_length() for extent in table.shape)
    _batched_lookup_cost(table, control_count, selection_widths, target_widths)
    if erase_logs is None:
        erase_logs = tuple(_optimal_erase_log(extent) for extent in table.shape)
    if len(erase_logs) != len(table.shape) or any(
        type(log) is not int or not 0 <= log <= extent.bit_length() - 1
        for extent, log in zip(table.shape, erase_logs, strict=True)
    ):
        raise ValueError("Erasure block sizes must fit the original table axes")

    block = 1 << sum(erase_logs)
    original_rows = prod(table.shape)
    padded_shape = tuple(
        batch * forward_block
        for batch, forward_block in zip(
            table.batch_shape, table.block_shape, strict=True
        )
    )
    banks = sum(target_widths) * prod(table.block_shape)
    original_ports = control_count + banks + sum(selection_widths)
    padded_ports = (
        control_count
        + banks
        + sum((extent - 1).bit_length() for extent in padded_shape)
    )

    def toffoli(rows: int) -> int:
        return max(0, (rows + block - 1) // block + block - 4 + control_count)

    # The wrapper declares a static gate formula based on the unpadded shape.
    # Its width is obtained from the padded leaf and the wrapper signature.
    return QroamErasureCosts(
        ComponentCost(
            "QROAMCleanAdjointWrapper",
            max(3, original_ports, padded_ports),
            GateCost(toffoli=toffoli(original_rows)),
        ),
        ComponentCost(
            "QROAMCleanAdjoint",
            max(3, padded_ports),
            GateCost(toffoli=toffoli(prod(padded_shape))),
        ),
    )


def _optimal_erase_log(extent: int) -> int:
    """Minimize the source objective extent / block + block, preferring smaller ties."""
    low = (extent.bit_length() - 1) // 2
    smaller = 1 << low
    return low + int(2 * smaller * smaller < extent)
