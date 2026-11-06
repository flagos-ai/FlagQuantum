# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROAMClean and SelectSwapQROM shape costs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qroam_clean.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/select_swap_qrom.py

"""Resource formulas for blocked lookups with shape metadata only."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import cast

from .._component_trace.symbolic_math import SymbolicInt, bit_length, ceil, log2, smax
from .qroam_resource_model import _optimal_erase_log
from .source_qroam_clean import optimal_qroam_clean_log_block_size
from .symbolic_resource_model import (
    SymbolicResourceFormula,
    source_qrom_resource_formula,
)


def _blocks(
    shape: tuple[SymbolicInt, ...], logs: tuple[SymbolicInt, ...]
) -> tuple[SymbolicInt, ...]:
    if not shape or len(shape) != len(logs):
        raise ValueError("Blocked lookup requires one exponent per nonempty shape axis")
    for extent, log in zip(shape, logs, strict=True):
        if isinstance(log, int) and (
            log < 0 or isinstance(extent, int) and 1 << log > extent
        ):
            raise ValueError("Lookup block sizes must fit the shape")
    return tuple(2**log for log in logs)


def _ceil_ratio(numerator: SymbolicInt, denominator: SymbolicInt) -> SymbolicInt:
    if isinstance(numerator, int) and isinstance(denominator, int):
        return (numerator + denominator - 1) // denominator
    return ceil(numerator / denominator)


def _forward_logs(
    shape: tuple[SymbolicInt, ...],
    target_widths: tuple[SymbolicInt, ...],
    dirty: bool | None,
) -> tuple[SymbolicInt, ...]:
    weight = sum(target_widths)
    if dirty is not None:
        weight *= len(target_widths) * (1 if dirty else 2)
    logs = []
    for extent in shape:
        if isinstance(extent, int) and isinstance(weight, int):
            exponent = optimal_qroam_clean_log_block_size(extent, weight)
        else:
            expression = 0.5 * log2(extent / weight)
            exponent = (
                cast(SymbolicInt, expression) if dirty is None else ceil(expression)
            )
        logs.append(exponent)
    return tuple(logs)


def source_blocked_lookup_resource_formula(
    shape: tuple[SymbolicInt, ...],
    target_widths: tuple[SymbolicInt, ...],
    log_block_sizes: tuple[SymbolicInt, ...] | None = None,
    *,
    control_count: SymbolicInt = 0,
    selection_widths: tuple[SymbolicInt, ...] | None = None,
    dirty: bool | None = None,
) -> SymbolicResourceFormula:
    """Model QROAMClean or SelectSwapQROM for declared shapes without data.

    The default selects QROAMClean. Specify dirty=False or True for the
    clean-bank or borrowed-bank SelectSwapQROM XOR lookup, respectively.
    """
    if dirty is not None and type(dirty) is not bool:
        raise ValueError("Select-swap dirty mode must be boolean")
    if not target_widths:
        raise ValueError("A lookup formula requires at least one target")
    if log_block_sizes is None:
        log_block_sizes = _forward_logs(shape, target_widths, dirty)
    blocks = _blocks(shape, log_block_sizes)
    selectors = (
        tuple(bit_length(extent - 1) for extent in shape)
        if selection_widths is None
        else selection_widths
    )
    if len(selectors) != len(shape):
        raise ValueError("Lookup selection widths must match the shape")
    high = tuple(
        width - log for width, log in zip(selectors, log_block_sizes, strict=True)
    )
    batch_shape = tuple(
        _ceil_ratio(extent, block) for extent, block in zip(shape, blocks, strict=True)
    )
    qrom = source_qrom_resource_formula(
        batch_shape,
        target_widths,
        target_shapes=(blocks,) * len(target_widths),
        control_count=control_count,
        selection_widths=high,
    )
    swaps = prod(blocks) - 1
    target_width = sum(target_widths)
    swap_t = swaps * 4 * target_width
    swap_clifford = swaps * (22 * target_width - len(target_widths))
    width = qrom.logical_qubits + sum(log_block_sizes)
    if dirty is None:
        return SymbolicResourceFormula(
            qrom.and_bloq,
            qrom.clifford + swap_clifford,
            qrom.measurement,
            width,
            t=swap_t,
        )
    copies = 1 + dirty
    return SymbolicResourceFormula(
        2 * qrom.and_bloq,
        2 * qrom.clifford + copies * (target_width + 2 * swap_clifford),
        2 * qrom.measurement,
        width + target_width,
        t=2 * copies * swap_t,
    )


@dataclass(frozen=True)
class ShapeErasureResources:
    """Separate width formulas for the source wrapper and opaque erase leaf."""

    wrapper: SymbolicResourceFormula
    leaf: SymbolicResourceFormula


def source_shape_erasure_resource_formula(
    shape: tuple[SymbolicInt, ...],
    target_widths: tuple[SymbolicInt, ...],
    forward_logs: tuple[SymbolicInt, ...] | None = None,
    *,
    control_count: SymbolicInt = 0,
    erase_logs: tuple[SymbolicInt, ...] | None = None,
) -> ShapeErasureResources:
    """Model the shape-only QROAM erase wrapper and leaf on the original shape."""
    if not target_widths:
        raise ValueError("An erasure formula requires at least one target")
    if forward_logs is None:
        forward_logs = _forward_logs(shape, target_widths, None)
    forward = _blocks(shape, forward_logs)
    if erase_logs is None:
        erase_logs = tuple(
            (
                _optimal_erase_log(extent)
                if isinstance(extent, int)
                else cast(SymbolicInt, 0.5 * log2(extent))
            )
            for extent in shape
        )
    erase = prod(_blocks(shape, erase_logs))
    toffoli = _ceil_ratio(prod(shape), erase) + erase - 4 + control_count
    if isinstance(toffoli, int):
        toffoli = max(0, toffoli)
    address = control_count + sum(bit_length(extent - 1) for extent in shape)
    blocks = prod(forward)
    ports = address + sum(width * blocks for width in target_widths)
    wrapper_ports = address + sum(
        width + width * (blocks - 1) for width in target_widths
    )
    return ShapeErasureResources(
        SymbolicResourceFormula(
            0, 0, 0, cast(SymbolicInt, smax(3, address, wrapper_ports)), toffoli=toffoli
        ),
        SymbolicResourceFormula(
            0, 0, 0, cast(SymbolicInt, smax(3, ports)), toffoli=toffoli
        ),
    )
