# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SingleWindowModMul.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Single Montgomery window using source arithmetic and explicit output aliases."""

from ...core.ir import Instruction
from .arithmetic import add_adjoint_ir, add_ir
from .bitwise import _validate_wires, xor_ir
from .controlled_addition import controlled_add_adjoint_ir, controlled_add_ir
from .source_qroam_clean import (
    QroamCleanTable,
    optimal_qroam_clean_log_block_size,
    source_qroam_clean_ir,
    source_qroam_clean_uncompute_ir,
)


def single_window_qroam_table(width: int, window: int, modulus: int) -> QroamCleanTable:
    """Build the original concrete correction dataset and forward block choice."""
    if (
        type(width) is not int
        or type(window) is not int
        or not 1 <= window <= width
        or width % window
    ):
        raise ValueError(
            "Montgomery window must be a positive divisor of operand width"
        )
    if type(modulus) is not int or not 1 <= modulus < 1 << width or modulus % 2 == 0:
        raise ValueError(
            "Montgomery window requires a positive odd modulus fitting the operand"
        )
    size = 1 << window
    inverse = pow(modulus, -1, size)
    values = tuple((-index * inverse % size) * modulus for index in range(size))
    log = optimal_qroam_clean_log_block_size(size, width + window)
    return QroamCleanTable((size,), (log,), (values,))


def single_window_mod_mul_ir(
    x: tuple[int, ...],
    y: tuple[int, ...],
    target: tuple[int, ...],
    index: tuple[int, ...],
    modulus: int,
    bank: tuple[tuple[int, ...], ...],
    work: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[tuple[Instruction, ...], tuple[int, ...]]:
    """Emit one source window and return the logical target wire order.

    x is little-endian; other words are big-endian. The w-bit index starts
    at zero in the forward direction and is a retained output. Forward target
    rotation is an alias. For the adjoint, supply that rotated input order.
    The integer recurrence applies on the carry-safe domain, for example
    2*p-1 < 2**n, y < p and target < 2*p. The gates retain source wraparound.

    Supply n+w-1 work wires, 2n classical slots and the clean lookup bank.
    Controlled addition borrows vacant bank wires. Bank and work return to
    zero. The executable bank clear reverses the forward lookup components.
    """
    n, w = len(y), len(x)
    table = single_window_qroam_table(n, w, modulus)
    if len(target) != n + w or len(index) != w or type(adjoint) is not bool:
        raise ValueError("Montgomery window requires n+w target bits and w index bits")
    if len(bank) != table.block_shape[0] or any(len(word) != n + w for word in bank):
        raise ValueError(
            "Montgomery window requires one n+w bit word per lookup bank cell"
        )
    if len(work) != n + w - 1 or len(classical_bits) != 2 * n:
        raise ValueError(
            "Montgomery window requires n+w-1 work wires and 2n classical slots"
        )
    bank_wires = tuple(wire for word in bank for wire in word)
    _validate_wires((*x, *y, *target, *index, *bank_wires, *work))

    accumulator = (*target[w:], *target[:w]) if adjoint else target
    output = accumulator if adjoint else (*target[-w:], *target[:-w])
    pool = (*work, *bank_wires)[: 2 * n]
    emit = controlled_add_adjoint_ir if adjoint else controlled_add_ir
    positions = reversed(range(w)) if adjoint else range(w)
    partial_products = tuple(
        gate
        for i in positions
        for gate in emit(
            x[i],
            y,
            accumulator[w - i - 1 : n + w - i],
            pool[:n],
            pool[n:],
            classical_bits[:n],
            classical_bits[n:],
        )
    )
    copy_index = xor_ir(accumulator[-w:], index)
    lookup_work = table.workspace_size(0, (w,))
    lookup_args = (
        table,
        (),
        (index,),
        (bank,),
        work[:lookup_work],
        classical_bits[:lookup_work],
    )
    lookup = source_qroam_clean_ir(*lookup_args)
    clear = source_qroam_clean_uncompute_ir(*lookup_args)
    add = add_adjoint_ir if adjoint else add_ir
    correction = add(bank[0], accumulator, work, classical_bits[: len(work)])
    middle = (*lookup, *correction, *clear)
    gates = (
        (*middle, *copy_index, *partial_products)
        if adjoint
        else (*partial_products, *copy_index, *middle)
    )
    return gates, output
