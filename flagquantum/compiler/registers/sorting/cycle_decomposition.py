# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 linalg/permutation.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/permutation.py

"""Source-order cycle discovery, separate from register validation and emission."""

from collections.abc import Iterator, Mapping, Sequence


def permutation_cycles(permutation: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """Yield nontrivial cycles through the original NetworkX call chain.

    Import NetworkX only when iteration starts. Graph node insertion order,
    simple-cycle discovery and invalid-input behavior belong to that provider;
    do not normalize rotations, sort cycles or validate bijectivity here.
    No physical wires or gates are allocated by cycle discovery.
    """
    import networkx as nx

    graph = nx.DiGraph(enumerate(permutation))
    for cycle in nx.simple_cycles(graph):
        if len(cycle) >= 2:
            yield tuple(cycle)


def partial_permutation_cycles(
    mapping: Mapping[int, int],
) -> Iterator[tuple[int, ...]]:
    """Yield original ordered paths without validating or completing the mapping.

    A missing outgoing edge or an already seen node ends a path. Prefixes that
    reach a previous path remain separate. Mapping iteration order is retained;
    callers requiring a valid partial permutation must enforce their own domain.
    The seen set and current path use O(number of visited nodes) storage.
    """
    yield from _partial_cycles(mapping)


def complete_injection_cycles(mapping: Mapping[int, int]) -> Iterator[tuple[int, ...]]:
    """Close each path of a validated injection from its actual head."""
    yield from _partial_cycles(
        mapping, {target: source for source, target in mapping.items()}
    )


def _partial_cycles(
    mapping: Mapping[int, int], predecessors: Mapping[int, int] | None = None
) -> Iterator[tuple[int, ...]]:
    seen: set[int] = set()
    for node in mapping:
        if node in seen:
            continue
        if predecessors is not None:
            start = node
            while node in predecessors:
                node = predecessors[node]
                if node == start:
                    break
        cycle = []
        while node not in seen:
            seen.add(node)
            cycle.append(node)
            if node not in mapping:
                break
            node = mapping[node]
        if len(cycle) >= 2:
            yield tuple(cycle)
