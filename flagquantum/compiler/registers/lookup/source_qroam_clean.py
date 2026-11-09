# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROAMClean and SelectSwapQROM.
# Sources:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qroam_clean.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/select_swap_qrom.py

"""Clean-bank lookup with source batch ordering and relative-phase routing."""

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import ceil, floor, log2, prod

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from .source_qrom import source_qrom_ir, source_qrom_workspace_size
from .swap_with_zero import swap_with_zero_ir


def optimal_qroam_clean_log_block_size(data_size: int, target_bits: int) -> int:
    """Choose the original concrete forward block exponent."""
    if (
        type(data_size) is not int
        or data_size < 1
        or type(target_bits) is not int
        or target_bits < 1
    ):
        raise ValueError(
            "QROAM block selection requires positive data and target sizes"
        )
    estimate = 0.5 * log2(data_size / target_bits)
    if estimate < 0:
        return 0

    def cost(log: int) -> float:
        block = 1 << log
        return data_size / block + target_bits * (block - 1)

    return min((floor(estimate), ceil(estimate)), key=cost)


@dataclass(frozen=True)
class QroamCleanTable:
    """Concrete scalar datasets on shared axes with explicit block exponents."""

    shape: tuple[int, ...]
    log_block_sizes: tuple[int, ...]
    datasets: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.shape or len(self.shape) != len(self.log_block_sizes):
            raise ValueError(
                "QROAM requires one block exponent per nonempty shape axis"
            )
        for extent, log in zip(self.shape, self.log_block_sizes, strict=True):
            if (
                type(extent) is not int
                or extent < 1
                or type(log) is not int
                or not 0 <= log <= extent.bit_length() - 1
            ):
                raise ValueError("QROAM block sizes must fit their axes")
        if not self.datasets or any(
            len(data) != prod(self.shape) for data in self.datasets
        ):
            raise ValueError("QROAM requires datasets matching the full table shape")
        if any(self.log_block_sizes) and (
            not all(self.log_block_sizes) or 1 in self.batch_shape
        ):
            raise ValueError(
                "Source blocked QROAM cannot bind omitted high or low selectors"
            )

    @cached_property
    def block_shape(self) -> tuple[int, ...]:
        return tuple(1 << log for log in self.log_block_sizes)

    @cached_property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(
            (extent + block - 1) // block
            for extent, block in zip(self.shape, self.block_shape, strict=True)
        )

    @cached_property
    def batched_data(self) -> tuple[tuple[int, ...], ...]:
        """Return each dataset in high-axes then low-axes order, padding with zero."""
        strides = tuple(prod(self.shape[i + 1 :]) for i in range(len(self.shape)))
        low_rows = tuple(product(*(range(block) for block in self.block_shape)))
        rows: list[int | None] = []
        for high in product(*(range(extent) for extent in self.batch_shape)):
            for low in low_rows:
                index = tuple(
                    major * block + minor
                    for major, block, minor in zip(
                        high, self.block_shape, low, strict=True
                    )
                )
                if any(
                    i >= extent for i, extent in zip(index, self.shape, strict=True)
                ):
                    rows.append(None)
                else:
                    rows.append(
                        sum(
                            i * stride for i, stride in zip(index, strides, strict=True)
                        )
                    )
        return tuple(
            tuple(data[row] if row is not None else 0 for row in rows)
            for data in self.datasets
        )

    def _high_widths(self, widths: tuple[int, ...]) -> tuple[int, ...]:
        if len(widths) != len(self.shape) or any(
            not (extent - 1).bit_length() <= width or (extent == 1 and width != 0)
            for extent, width in zip(self.shape, widths, strict=True)
        ):
            raise ValueError("QROAM selection widths must fit the source axes")
        return tuple(
            width - log for width, log in zip(widths, self.log_block_sizes, strict=True)
        )

    def workspace_size(
        self, control_count: int, selection_widths: tuple[int, ...]
    ) -> int:
        """Count temporary And qubits, excluding the output banks."""
        return source_qrom_workspace_size(
            control_count, self._high_widths(selection_widths), self.batched_data
        )


def source_qroam_clean_ir(
    table: QroamCleanTable,
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Load clean banks with the original signed target and logical junk.

    Banks are zero-initialized and have one word per row-major block cell.
    The first word of each bank is the selected target. Remaining words are
    retained outputs. Only the caller-bound And workspace returns to zero.
    """
    return _emit_clean_qroam(
        table, controls, selections, banks, workspace, classical_bits, reverse=False
    )


def source_qroam_clean_uncompute_ir(
    table: QroamCleanTable,
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Reverse the forward stages on their signed output banks.

    Undo relative-phase routing before XORing the original batch load.
    The temporary And work starts and ends at zero. The target and junk
    words also end at zero when they carry the matching forward data.
    """
    return _emit_clean_qroam(
        table, controls, selections, banks, workspace, classical_bits, reverse=True
    )


def _emit_clean_qroam(
    table: QroamCleanTable,
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    reverse: bool,
) -> tuple[Instruction, ...]:
    load, low = _source_qroam_batch_ir(
        table, controls, selections, banks, workspace, classical_bits
    )
    route = _source_qroam_route_ir(table, low, banks, reverse=reverse)
    return (*route, *load) if reverse else (*load, *route)


def _source_qroam_route_ir(
    table: QroamCleanTable,
    low: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    reverse: bool,
) -> tuple[Instruction, ...]:
    if not any(table.log_block_sizes):
        return ()
    bank_order = reversed(banks) if reverse else iter(banks)
    return tuple(
        gate
        for bank in bank_order
        for gate in swap_with_zero_ir(low, table.block_shape, bank, adjoint=reverse)
    )


def _source_qroam_batch_ir(
    table: QroamCleanTable,
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    outer_control: int | None = None,
) -> tuple[tuple[Instruction, ...], tuple[tuple[int, ...], ...]]:
    """Validate the common batch layout and return its load and low selectors."""
    high_widths = table._high_widths(tuple(map(len, selections)))
    high = tuple(
        wires[:width] for wires, width in zip(selections, high_widths, strict=True)
    )
    low = tuple(
        wires[width:] for wires, width in zip(selections, high_widths, strict=True)
    )
    if any(len(bank) != prod(table.block_shape) for bank in banks):
        raise ValueError("QROAM requires one bank word per block cell")
    _validate_wires(
        (
            *controls,
            *(wire for axis in selections for wire in axis),
            *(wire for bank in banks for word in bank for wire in word),
            *workspace,
        )
    )
    load = source_qrom_ir(
        controls,
        high,
        table.batch_shape,
        banks,
        table.batched_data,
        workspace,
        classical_bits,
        outer_control=outer_control,
    )
    return load, low
