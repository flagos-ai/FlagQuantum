# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PermutationCycle costs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/permutation.py

"""Logical costs for ordered equality-driven basis permutations."""

from .._component_trace.resource_model import ComponentCost, CostCall, GateCost


def source_permutation_cost(
    size: int, width: int, cycles: tuple[tuple[int, ...], ...]
) -> ComponentCost:
    """Count source cycles in order, including the final equality of each cycle."""
    if (
        type(width) is not int
        or width < 1
        or type(size) is not int
        or not 1 <= size <= 1 << width
    ):
        raise ValueError("Permutation size must fit a nonempty register")
    calls = []
    for cycle in cycles:
        if (
            not cycle
            or len(set(cycle)) != len(cycle)
            or any(type(value) is not int or not 0 <= value < size for value in cycle)
        ):
            raise ValueError("Permutation cycle requires distinct entries in [0,size)")
        comparisons = (*cycle, cycle[0])
        flips = sum(
            (value ^ cycle[(i + 1) % len(cycle)]).bit_count()
            for i, value in enumerate(cycle)
        )
        prefix_ands = (width - 1) * len(comparisons)
        # MultiAnd's decomposed children include polarity flips. Two
        # controls use the separately classified direct And.
        per_zero = 2 if width == 1 else 4 if width > 2 else 0
        zero_bits = sum(width - value.bit_count() for value in comparisons)
        calls.append(
            CostCall(
                ComponentCost(
                    "PermutationCycle",
                    2 * width,
                    GateCost(
                        and_bloq=prefix_ands,
                        clifford=flips
                        + width * len(comparisons)
                        + per_zero * zero_bits,
                        measurement=prefix_ands,
                    ),
                )
            )
        )
    return ComponentCost("BasisPermutation", width, calls=tuple(calls))
