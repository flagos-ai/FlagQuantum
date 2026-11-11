# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/lcu_util.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/lcu_util.py

"""Source-ordered probability rounding and integer alias-table preprocessing."""

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import overload

from ...core.value_math import SymbolicInt, SymbolicReal, ceil, log2
from ...core.value_parameters import is_symbolic_value


def _partial_sums(values: Iterable[float]) -> Iterator[float]:
    total: float = 0
    for value in values:
        yield total
        total += value
    yield total


def _differences(values: Iterable[int]) -> Iterator[int]:
    previous = 0
    have_previous = False
    for value in values:
        if have_previous:
            yield value - previous
        previous = value
        have_previous = True


def discretize_probabilities(
    weights: Sequence[float], sub_bit_precision: int
) -> tuple[list[int], int]:
    """Round cumulative probabilities before differencing, as in the source.

    This raw boundary retains original errors and arithmetic, without the
    finite/nonnegative checks of the validated alias-table factory.
    """
    count = len(weights)
    denominator = 2**sub_bit_precision * count
    cumulative = list(_partial_sums(weights))
    total = cumulative[-1]
    rounded = [
        int(math.floor(value / total * denominator + 0.5)) for value in cumulative
    ]
    return list(_differences(rounded)), denominator


def integer_alias_rows(weights: Sequence[int]) -> tuple[list[int], list[int]]:
    """Assign source alternates and keep weights in two ordered donor passes.

    The input is copied. Its sum must be divisible by its nonzero length;
    nonnegativity is not checked at this raw source boundary.
    """
    bins = list(weights)
    if not bins:
        raise ValueError("Empty input.")
    count = len(bins)
    target = sum(bins) // count
    if sum(bins) != count * target:
        raise ValueError(
            f"sum(weights)={sum(bins)} must be a multiple of len(weights)={len(bins)}."
        )
    alternates = list(range(count))
    keep = [0] * count
    donor = 0
    for _ in range(2):
        for i in range(count):
            if bins[i] >= target:
                continue
            while bins[donor] <= target:
                donor += 1
            bins[donor] -= target - bins[i]
            alternates[i] = donor
            keep[i] = bins[i]
            bins[i] = target
    return alternates, keep


@overload
def alias_precision_bits(rows: int, precision: float) -> int: ...


@overload
def alias_precision_bits(rows: SymbolicInt, precision: SymbolicReal) -> SymbolicInt: ...


def alias_precision_bits(rows: SymbolicInt, precision: SymbolicReal) -> SymbolicInt:
    """Retain the source formula, concrete one-bit floor and symbolic result."""
    bits = ceil(log2(1 / (precision * rows)))
    if not is_symbolic_value(bits) and bits < 1:
        return 1
    return bits


def reversible_sampling_rows(
    weights: Sequence[float],
    epsilon: float | None = None,
    sub_bit_precision: int | None = None,
) -> tuple[list[int], list[int], int]:
    """Prepare raw alias rows from exactly one normalized tolerance or width.

    Preserve the source checks, failure order and returned mutable lists. This
    helper emits no gates and owns no quantum work or runtime sampling state.
    """
    if not ((epsilon is None) ^ (sub_bit_precision is None)):
        raise ValueError("Exactly one of epsilon or sub_bit_prec must be provided")
    if sub_bit_precision is None:
        assert epsilon is not None
        sub_bit_precision = alias_precision_bits(len(weights), epsilon)
    if sub_bit_precision <= 0:
        raise ValueError(f"{sub_bit_precision=} must be at least 1")
    numerators, denominator = discretize_probabilities(weights, sub_bit_precision)
    assert denominator == 2**sub_bit_precision * len(numerators)
    alternates, keep = integer_alias_rows(numerators)
    return alternates, keep, sub_bit_precision
