"""Restore tokens using real device edges while preserving physical ownership."""

from __future__ import annotations

from collections import deque
from itertools import combinations, permutations

import pytest

from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.compiler.topology.sabre import restore_swaps

pytestmark = pytest.mark.unit


def _reachable_tokens(graph: CouplingMap) -> dict[tuple[int, ...], int]:
    """Enumerate independent minimum distances in the token-state graph."""
    identity = tuple(range(graph.n_qubits))
    distances = {identity: 0}
    queue = deque([identity])
    while queue:
        tokens = queue.popleft()
        for left, right in graph.edges:
            following = list(tokens)
            following[left], following[right] = following[right], following[left]
            state = tuple(following)
            if state not in distances:
                distances[state] = distances[tokens] + 1
                queue.append(state)
    return distances


def _replay(
    tokens: tuple[int, ...], graph: CouplingMap, swaps: tuple[tuple[int, int], ...]
) -> None:
    occupants = list(tokens)
    for left, right in swaps:
        assert 0 <= left < right < len(tokens)
        assert graph.has_edge(left, right)
        occupants[left], occupants[right] = occupants[right], occupants[left]
    assert occupants == list(range(len(tokens)))


def test_all_four_wire_graphs_and_permutations_against_token_state_oracle() -> None:
    edges = tuple(combinations(range(4), 2))
    for mask in range(1 << len(edges)):
        graph = CouplingMap(
            4, tuple(edge for index, edge in enumerate(edges) if mask & (1 << index))
        )
        reachable = _reachable_tokens(graph)
        for tokens in permutations(range(4)):
            layout = tuple(tokens.index(logical) for logical in range(4))
            if tokens not in reachable:
                with pytest.raises(ValueError, match="no coupling path"):
                    restore_swaps(layout, graph)
                continue
            swaps = restore_swaps(layout, graph)
            _replay(tokens, graph, swaps)
            assert len(swaps) >= reachable[tokens]
            assert restore_swaps(layout, graph) == swaps


def test_adjacent_tokens_use_ring_edge_missing_from_spanning_tree() -> None:
    graph = CouplingMap.ring(6)
    layout = (0, 1, 2, 4, 3, 5)
    swaps = restore_swaps(layout, graph)
    # The existing tree walk takes nine SWAPs around the other side of the ring.
    assert swaps == ((3, 4),)
    _replay(layout, graph, swaps)


def test_clique_restoration_uses_minimum_transpositions() -> None:
    graph = CouplingMap(5, combinations(range(5), 2))
    distances = _reachable_tokens(graph)
    for tokens, minimum in distances.items():
        layout = tuple(tokens.index(logical) for logical in range(5))
        swaps = restore_swaps(layout, graph)
        _replay(tokens, graph, swaps)
        assert len(swaps) == minimum


def test_cycles_do_not_allow_transit_through_unused_device_qubits() -> None:
    graph = CouplingMap(5, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)))
    layout = (2, 1, 0)
    swaps = restore_swaps(layout, graph)
    assert len(swaps) == 3
    _replay(layout, graph, swaps)


def test_separate_cyclic_components_restore_without_crossing() -> None:
    graph = CouplingMap(6, ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)))
    layout = (2, 0, 1, 4, 5, 3)
    tokens = tuple(layout.index(physical) for physical in range(6))
    swaps = restore_swaps(layout, graph)
    _replay(tokens, graph, swaps)
    assert len(swaps) == 4
