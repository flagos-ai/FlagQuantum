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
from ...core.ir.program.signatures import Signature


def chemistry_qroam_signature(rows: int, width: int) -> Signature:
    """Preserve the selection and target ports of both source placeholders."""
    if any(type(n) is not int or n < 1 for n in (rows, width)):
        raise ValueError("Chemistry QROAM requires positive row and target widths")
    return Signature.build(sel=(rows - 1).bit_length(), trg=width)


def chemistry_qroam_block_score(
    rows: int,
    width: int,
    block: int,
    *,
    adjoint: bool = False,
    dirty: bool = False,
) -> float:
    """Evaluate the original unrounded clean or dirty scalar block score."""
    if adjoint:
        return rows / block + block
    if dirty:
        return 2 * (rows / block - 1) + 4 * width * (block - 1)
    return rows / block + width * (block - 1)


def chemistry_qroam_log_block_size(
    rows: int, width: int, *, block_size: int | None = None, adjoint: bool = False
) -> int:
    """Select the original clean chemistry block exponent in either direction."""
    chemistry_qroam_signature(rows, width)
    if block_size is not None and (type(block_size) is not int or block_size < 1):
        raise ValueError("Chemistry QROAM requires a positive block size")
    if block_size is None and not adjoint:
        if rows < width:
            raise ValueError(
                "Source chemistry QROAM block selection requires rows >= target width"
            )
        return optimal_qroam_clean_log_block_size(rows, width)
    ideal = log2(block_size) if block_size is not None else log2(rows) / 2

    def score(exponent: int) -> float:
        return chemistry_qroam_block_score(rows, width, 1 << exponent, adjoint=adjoint)

    return min((floor(ideal), ceil(ideal)), key=score)


def source_chemistry_qroam_cost(
    rows: int, width: int, *, block_size: int | None = None, adjoint: bool = False
) -> ComponentCost:
    """Use the placeholder's optimal block or round an explicit block to a power of two."""
    exponent = chemistry_qroam_log_block_size(
        rows, width, block_size=block_size, adjoint=adjoint
    )
    if block_size == 1:
        cost = rows - 1
    else:
        block = 1 << exponent
        cost = (rows + block - 1) // block + (block if adjoint else width * (block - 1))
    signature = chemistry_qroam_signature(rows, width)
    # Original QubitCount falls back to max(signature, Toffoli arity), even
    # when the declared Toffoli multiplicity is zero. This is a model peak,
    # not a private workspace allocation or an executable lookup width.
    return ComponentCost(
        "QROAM", max(3, int(signature.n_qubits())), GateCost(toffoli=cost)
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
    signature = chemistry_qroam_signature(first, width)
    return ComponentCost(
        "QROAMTwoRegs", max(3, int(signature.n_qubits())), GateCost(toffoli=cost)
    )
