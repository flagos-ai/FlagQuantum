# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectSwapQROM.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/select_swap_qrom.py

"""Select a QROM word by XOR while restoring clean or borrowed banks."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, xor_ir
from .source_qroam_clean import (
    QroamCleanTable,
    _source_qroam_batch_ir,
    optimal_qroam_clean_log_block_size,
)
from .swap_with_zero import swap_with_zero_ir


def optimal_select_swap_log_block_sizes(
    shape: tuple[int, ...], target_widths: tuple[int, ...], *, dirty: bool = True
) -> tuple[int, ...]:
    """Return the original concrete scalar-dataset block cost minimizer."""
    if (
        type(dirty) is not bool
        or not shape
        or not target_widths
        or any(type(width) is not int or width < 1 for width in target_widths)
    ):
        raise ValueError(
            "Select-swap block selection requires a shape and positive widths"
        )
    # Preserve the implemented cost rule and scalar target-shape factor.
    weight = sum(target_widths) * len(target_widths) * (1 if dirty else 2)
    return tuple(optimal_qroam_clean_log_block_size(extent, weight) for extent in shape)


def select_swap_qrom_ir(
    table: QroamCleanTable,
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    targets: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    dirty: bool = True,
) -> tuple[Instruction, ...]:
    """XOR each selected word into its destination, restoring the bank state.

    Addresses and words are big-endian. The valid address range includes the
    padded cells of the final high block. Every destination can start nonzero.
    In the dirty path each bank can carry arbitrary coherent data, including
    correlations with other registers. In the clean path banks start at zero.
    Measured And workspace starts and ends at zero in both cases.
    """
    if type(dirty) is not bool:
        raise ValueError("Select-swap dirty mode must be boolean")
    load, low = _source_qroam_batch_ir(
        table, controls, selections, banks, workspace, classical_bits
    )
    if len(targets) != len(banks) or any(
        len(target) != len(bank[0]) for target, bank in zip(targets, banks, strict=True)
    ):
        raise ValueError("Select-swap requires one matching destination per bank")
    _validate_wires(
        (
            *controls,
            *(wire for axis in selections for wire in axis),
            *(wire for bank in banks for word in bank for wire in word),
            *(wire for word in targets for wire in word),
            *workspace,
        )
    )

    copy = tuple(
        gate
        for bank, target in zip(banks, targets, strict=True)
        for gate in xor_ir(bank[0], target)
    )
    route: list[Instruction] = []
    unroute: list[Instruction] = []
    if any(table.log_block_sizes):
        for bank in banks:
            route.extend(swap_with_zero_ir(low, table.block_shape, bank))
            unroute.extend(
                swap_with_zero_ir(low, table.block_shape, bank, adjoint=True)
            )
    select = (*route, *copy, *unroute)
    return (*load, *select, *load, *select) if dirty else (*load, *select, *load)
