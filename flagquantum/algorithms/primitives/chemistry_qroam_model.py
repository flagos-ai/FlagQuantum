# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry QROAM cost placeholders.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/black_boxes.py

"""Analytical chemistry QROAM costs for one and two address axes."""

from math import ceil, floor, log2

from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler.registers.lookup.source_qroam_clean import (
    optimal_qroam_clean_log_block_size,
)


def source_chemistry_qroam_cost(
    rows: int, width: int, *, block_size: int | None = None, adjoint: bool = False
) -> ComponentCost:
    """Use the placeholder's optimal block or round an explicit block to a power of two."""
    if any(type(n) is not int or n < 1 for n in (rows, width)):
        raise ValueError("Chemistry QROAM requires positive row and target widths")
    if block_size is not None and (type(block_size) is not int or block_size < 1):
        raise ValueError("Chemistry QROAM requires a positive block size")
    if block_size == 1:
        cost = rows - 1
    else:
        if block_size is None and not adjoint:
            if rows < width:
                raise ValueError(
                    "Source chemistry QROAM block selection requires rows >= target width"
                )
            exponent = optimal_qroam_clean_log_block_size(rows, width)
        else:
            ideal = log2(block_size) if block_size is not None else log2(rows) / 2

            def score(exponent: int) -> float:
                block = 1 << exponent
                return rows / block + (block if adjoint else width * (block - 1))

            exponent = min((floor(ideal), ceil(ideal)), key=score)
        block = 1 << exponent
        cost = (rows + block - 1) // block + (block if adjoint else width * (block - 1))
    return ComponentCost(
        "QROAM", (rows - 1).bit_length() + width, GateCost(toffoli=cost)
    )


def source_chemistry_qroam_two_axis_cost(
    first: int,
    second: int,
    first_block: int,
    second_block: int,
    width: int,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Keep both model axes and the original first-axis-only signature width."""
    if any(
        type(n) is not int or n < 1
        for n in (first, second, first_block, second_block, width)
    ):
        raise ValueError("Two-axis chemistry QROAM requires positive dimensions")
    blocks = ((first + first_block - 1) // first_block) * (
        (second + second_block - 1) // second_block
    )
    bank = first_block * second_block
    cost = blocks + (bank if adjoint else width * (bank - 1))
    return ComponentCost(
        "QROAMTwoRegs", (first - 1).bit_length() + width, GateCost(toffoli=cost)
    )
