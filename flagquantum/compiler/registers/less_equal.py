# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned Qualtran v0.7.0 LessThanEqual and _equality_with_zero.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR, shared components and explicit work/measurement ownership.

"""Unsigned register less-or-equal with the source comparison tree."""

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost, GateCost
from .and_compute import and_compute_ir, and_uncompute_ir
from .comparison_tree import bi_qubits_mixer_ir, single_qubit_compare_ir
from .multi_control import multi_and_ir


def less_equal_requirements(left_width: int, right_width: int) -> tuple[int, int]:
    """Return (clean quantum qubits, distinct classical slots) for concrete widths."""
    if any(type(n) is not int or n < 1 for n in (left_width, right_width)):
        raise ValueError("Less-equal requires positive concrete register widths")
    n = min(left_width, right_width)
    d = abs(left_width - right_width)
    return (
        3 * n - 1 + (max(1, d - 1) + 1 if d else 0),
        2 * n - 1 + (max(0, d - 1) + 1 if d else 0),
    )


def source_equal_width_less_equal_cost(width: int) -> ComponentCost:
    """Count the source comparison tree for two equal-width concrete words."""
    work, slots = less_equal_requirements(width, width)
    return ComponentCost(
        "LessThanEqual",
        2 * width + 1 + work,
        GateCost(and_bloq=slots, clifford=13 + 22 * (width - 1), measurement=slots),
    )


def _tree(
    x: tuple[int, ...], y: tuple[int, ...]
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if len(x) == 1:
        return []
    if len(x) == 2:
        return [(x, y)]
    m = len(x) // 2
    return [
        *_tree(x[:m], y[:m]),
        *_tree(x[m:], y[m:]),
        ((x[m - 1], x[-1]), (y[m - 1], y[-1])),
    ]


def less_equal_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR unsigned a<=b into any target, preserving inputs and clearing all work.

    Registers are big-endian and may have different positive widths. The source
    prefix-zero cases and comparison tree are kept intact. Caller-owned work is
    initially zero; intermediate component flags are logical outputs until the
    recorded reverse cleanup. Use less_equal_requirements for both resource sizes.
    """
    n_work, n_bits = less_equal_requirements(len(left), len(right))
    wires = (*left, *right, target, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Less-equal requires distinct nonnegative quantum wires")
    if len(workspace) != n_work:
        raise ValueError(f"Less-equal requires {n_work} clean work wires")
    if (
        len(classical_bits) != n_bits
        or len(set(classical_bits)) != n_bits
        or any(type(b) is not int or b < 0 for b in classical_bits)
    ):
        raise ValueError(
            f"Less-equal requires {n_bits} distinct nonnegative classical slots"
        )
    cursor = 0
    slots = iter(classical_bits)

    def take(count: int) -> tuple[int, ...]:
        nonlocal cursor
        result = workspace[cursor : cursor + count]
        cursor += count
        return result

    gates: list[Instruction] = []
    cleanup: list[tuple[Instruction, ...]] = []
    n = min(len(left), len(right))
    d = abs(len(left) - len(right))
    prefix_equal: int | None = None
    if d:
        (prefix_equal,) = take(1)
        prefix = left[:-n] if len(left) > len(right) else right[:-n]
        if d == 1:
            block = (
                Instruction("x", prefix),
                Instruction("cx", (prefix[0], prefix_equal)),
            )
            gates.extend(block)
            cleanup.append(tuple(reversed(block)))
        elif d == 2:
            gates.extend(
                and_compute_ir(prefix[0], prefix[1], prefix_equal, values=(0, 0))
            )
            cleanup.append(
                and_uncompute_ir(
                    prefix[0], prefix[1], prefix_equal, next(slots), values=(0, 0)
                )
            )
        else:
            junk = take(d - 2)
            outputs = (*junk, prefix_equal)
            bits = tuple(next(slots) for _ in outputs)
            gates.extend(multi_and_ir(prefix, (0,) * d, outputs))
            cleanup.append(
                multi_and_ir(
                    prefix, (0,) * d, outputs, uncompute=True, classical_bits=bits
                )
            )
        if len(right) > len(left):
            gates.extend(
                (Instruction("x", (target,)), Instruction("cx", (prefix_equal, target)))
            )
    suffix_left, suffix_right = left[-n:], right[-n:]
    for x, y in _tree(suffix_left, suffix_right):
        aux = take(3)
        bits = (next(slots), next(slots))
        gates.extend(bi_qubits_mixer_ir(x, y, aux))
        cleanup.append(bi_qubits_mixer_ir(x, y, aux, adjoint=True, classical_bits=bits))
    less, greater = take(2)
    gates.extend(
        single_qubit_compare_ir(suffix_left[-1], suffix_right[-1], less, greater)
    )
    cleanup.append(
        single_qubit_compare_ir(
            suffix_left[-1],
            suffix_right[-1],
            less,
            greater,
            adjoint=True,
            classical_bits=(next(slots),),
        )
    )
    if prefix_equal is None:
        gates.extend(
            (Instruction("x", (target,)), Instruction("cx", (greater, target)))
        )
    else:
        (combined,) = take(1)
        gates.extend(and_compute_ir(prefix_equal, greater, combined, values=(1, 0)))
        cleanup.append(
            and_uncompute_ir(
                prefix_equal, greater, combined, next(slots), values=(1, 0)
            )
        )
        gates.append(Instruction("cx", (combined, target)))
    for erase_block in reversed(cleanup):
        gates.extend(erase_block)
    return tuple(gates)
