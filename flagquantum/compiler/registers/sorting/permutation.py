# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PermutationCycle and explicit Permutation cycles.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/permutation.py
# Changes: explicit work/classical slots, shared register primitives and native IR.

"""Ordered basis-state permutation cycles with measured equality cleanup."""

from collections.abc import Mapping

from ....core.ir import Instruction
from ..bits.bitwise import controlled_xor_constant_ir
from ..comparison.equality import equality_constant_ir


def source_partial_permutation_cycles(
    size: int, mapping: Mapping[int, int]
) -> tuple[tuple[int, ...], ...]:
    """Read nontrivial cycles in the locked source's mapping iteration order.

    A path ends on a missing outgoing edge or an already visited vertex. A
    prefix reaching an earlier path does not merge into it. Callers requiring
    all partial edges must verify the resulting mapping.

    Source: Qualtran v0.7.0 qualtran/linalg/permutation.py.
    """
    if type(size) is not int or size < 1:
        raise ValueError("Permutation size must be a positive integer")
    if len(set(mapping.values())) != len(mapping) or any(
        type(value) is not int or not 0 <= value < size
        for edge in mapping.items()
        for value in edge
    ):
        raise ValueError("Partial permutation must be injective and in range")

    seen: set[int] = set()
    cycles = []
    for start in mapping:
        cycle = []
        node = start
        while node not in seen:
            seen.add(node)
            cycle.append(node)
            if node not in mapping:
                break
            node = mapping[node]
        if len(cycle) >= 2:
            cycles.append(tuple(cycle))
    return tuple(cycles)


def permutation_cycle_ir(
    target: tuple[int, ...],
    cycle: tuple[int, ...],
    marker: int,
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    size: int,
) -> tuple[Instruction, ...]:
    """Map each cycle entry to its successor and leave other basis states alone.

    Big-endian target has width n; 1 <= size <= 2**n and distinct cycle entries
    lie in [0,size). Supply one zero marker, n-1 zero carries and n-1 distinct
    classical slots. Reused slots record the most recent measurement, not all
    event history. All quantum work returns to zero at the end of each cycle.
    Preserve singleton-cycle work and the final equality; do not optimize them.
    """
    if not target or type(size) is not int or not 1 <= size <= 1 << len(target):
        raise ValueError("Permutation size must fit a nonempty target register")
    if (
        not cycle
        or len(set(cycle)) != len(cycle)
        or any(type(value) is not int or not 0 <= value < size for value in cycle)
    ):
        raise ValueError("Permutation cycle requires distinct entries in [0,size)")
    gates: list[Instruction] = []
    for i, value in enumerate(cycle):
        gates.extend(
            equality_constant_ir(target, value, marker, carries, classical_bits)
        )
        delta = value ^ cycle[(i + 1) % len(cycle)]
        gates.extend(controlled_xor_constant_ir(marker, target, delta))
    gates.extend(
        equality_constant_ir(target, cycle[0], marker, carries, classical_bits)
    )
    return tuple(gates)


def permutation_ir(
    target: tuple[int, ...],
    cycles: tuple[tuple[int, ...], ...],
    marker: int,
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    size: int,
) -> tuple[Instruction, ...]:
    """Apply an explicit nonempty cycle sequence in source order.

    Cross-cycle overlap is allowed: composition order is observable. Marker,
    carries and measurement slots are reused only after the prior cycle clears
    its quantum work. Dense/partial-map conversion and empty sequences are not
    this concrete interface; allocation remains caller-owned.
    """
    if not cycles:
        raise ValueError("Permutation requires a nonempty explicit cycle sequence")
    return tuple(
        gate
        for cycle in cycles
        for gate in permutation_cycle_ir(
            target, cycle, marker, carries, classical_bits, size=size
        )
    )
