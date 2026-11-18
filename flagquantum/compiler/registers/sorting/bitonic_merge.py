# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitonicMerge.build_composite_bloq and its adjoint.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py
# Changes: native IR, original ports/history order and one shared clean work pool.

"""Original bitonic merge composition over shared comparison primitives."""

from ....core.ir import Instruction
from ..._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    estimate_component_resources,
)
from ..._component_trace.symbolic_math import SymbolicInt, bit_length, smax
from ..comparison.compare_exchange import (
    compare_exchange_ir,
    source_compare_exchange_cost,
    source_compare_exchange_resource_formula,
)
from ..comparison.comparison_layout import validate_comparison_layout
from ..comparison.parallel_comparison import parallel_compare_ir
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def bitonic_merge_history_count(half_length: int) -> int:
    """Return original logical history length for a positive power-of-two half."""
    if (
        type(half_length) is not int
        or half_length < 1
        or half_length & (half_length - 1)
    ):
        raise ValueError("Bitonic merge half length must be a positive power of two")
    return half_length * ((half_length - 1).bit_length() + 1)


def bitonic_merge_ir(
    words: tuple[tuple[int, ...], ...],
    histories: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Merge two sorted halves, or restore a correct complete forward output.

    Words are equal-width big-endian unsigned registers; each half has positive
    power-of-two length h. Forward result ports are the original physical words
    in concatenated order. History starts zero and remains logical output;
    inverse requires correct result/history and clears it. Work starts/ends zero
    and classical slots retain latest outcomes. Unsorted halves execute the same
    source network but are not promised to produce globally sorted output.
    """
    if len(words) % 2:
        raise ValueError("Bitonic merge requires two equal-length halves")
    half = len(words) // 2
    count = bitonic_merge_history_count(half)
    validate_comparison_layout(
        words,
        histories,
        workspace,
        classical_bits,
        history_count=count,
        adjoint=adjoint,
    )
    indices = range(half - 1, -1, -1) if adjoint else range(half)
    first = tuple(
        gate
        for i in indices
        for gate in compare_exchange_ir(
            words[i],
            words[2 * half - 1 - i],
            histories[i],
            workspace,
            classical_bits,
            adjoint=adjoint,
        )
    )
    rounds = tuple(
        (1 << offset, histories[(j + 1) * half : (j + 2) * half])
        for j, offset in enumerate(reversed(range((half - 1).bit_length())))
    )
    rest = tuple(
        gate
        for offset, history in (reversed(rounds) if adjoint else rounds)
        for gate in parallel_compare_ir(
            words, offset, history, workspace, classical_bits, adjoint=adjoint
        )
    )
    return (*rest, *first) if adjoint else (*first, *rest)


def source_bitonic_merge_cost(
    half_length: int, width: int, *, adjoint: bool = False
) -> ComponentCost:
    """Preserve original comparator multiplicity and retained-history peak."""
    if type(half_length) is not int or half_length < 1:
        raise ValueError("Bitonic merge model requires a positive integer half length")
    count = half_length * ((half_length - 1).bit_length() + 1)
    compare = source_compare_exchange_cost(width, adjoint=adjoint)
    # At the last comparison, every other input and prior history remains live.
    bystanders = (
        2 * (half_length - 1) * width + count - 1
        if half_length & (half_length - 1) == 0
        else 0
    )
    return ComponentCost(
        "BitonicMergeAdjoint" if adjoint else "BitonicMerge",
        2 * half_length * width + count,
        calls=(CostCall(compare, count, bystanders),),
    )


def source_bitonic_merge_resource_formula(
    half_length: SymbolicInt, width: SymbolicInt
) -> SymbolicResourceFormula:
    """Retain composite peaks for bound lengths and fallback for symbolic lengths."""
    if type(half_length) is int and type(width) is int:
        report = estimate_component_resources(
            source_bitonic_merge_cost(half_length, width)
        )
        return SymbolicResourceFormula(
            report.gates.and_bloq,
            report.gates.clifford,
            report.gates.measurement,
            report.peak_logical_qubits,
            cswap=report.gates.cswap,
        )
    comparison = source_compare_exchange_resource_formula(width)
    count = half_length * (bit_length(half_length - 1) + 1)
    ports = 2 * half_length * width + count
    if (
        type(half_length) is int
        and half_length > 0
        and half_length & (half_length - 1) == 0
    ):
        first_round_peak = (
            2 * (half_length - 1) * width + half_length - 1 + comparison.logical_qubits
        )
        parallel_peak = smax(2 * half_length * width + half_length, first_round_peak)
        peak = smax(ports, parallel_peak + count - half_length, first_round_peak)
    else:
        peak = smax(ports, 2 * half_length * width, comparison.logical_qubits)
    return SymbolicResourceFormula(
        count * comparison.and_bloq,
        count * comparison.clifford,
        count * comparison.measurement,
        peak,
        cswap=count * width,
    )
