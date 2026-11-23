"""Guard restored layouts against delayed input readiness and lost parallelism."""

from __future__ import annotations

from itertools import combinations, permutations, product
from unittest.mock import patch

import pytest

from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology.coupling import CouplingMap

pytestmark = pytest.mark.unit


def _finish(
    swaps: tuple[tuple[int, int], ...], arrivals: tuple[int, ...]
) -> tuple[int, ...]:
    # An independent clock simulation of unit-duration two-wire operations.
    ready = list(arrivals)
    for left, right in swaps:
        start = max(ready[left], ready[right])
        ready[left] = start + 1
        ready[right] = start + 1
    return tuple(ready)


def test_max_plus_guard_matches_all_small_clock_schedules() -> None:
    edges = tuple(combinations(range(3), 2))
    sequences = tuple(
        sequence for length in range(4) for sequence in product(edges, repeat=length)
    )
    arrivals = tuple(product((0, 1, 7), repeat=3))
    # Seven exceeds either sequence's possible internal delay, so it also
    # exposes missing dependencies hidden when every input starts at zero.
    results = {
        sequence: tuple(_finish(sequence, a) for a in arrivals)
        for sequence in sequences
    }
    for candidate, baseline in product(sequences, repeat=2):
        expected = all(
            all(new <= old for new, old in zip(new_times, old_times, strict=True))
            for new_times, old_times in zip(
                results[candidate], results[baseline], strict=True
            )
        )
        assert sabre._restore_no_later(candidate, baseline, 3) is expected


def test_fewer_swaps_can_delay_a_wire_even_for_the_same_permutation() -> None:
    candidate = ((1, 2), (0, 2))
    baseline = ((0, 1), (0, 1), (0, 1), (1, 2))

    def occupants(swaps: tuple[tuple[int, int], ...]) -> list[int]:
        tokens = list(range(3))
        for left, right in swaps:
            tokens[left], tokens[right] = tokens[right], tokens[left]
        return tokens

    assert occupants(candidate) == occupants(baseline)
    assert len(candidate) < len(baseline)
    assert not sabre._restore_no_later(candidate, baseline, 3)
    assert _finish(candidate, (0, 0, 10))[0] > _finish(baseline, (0, 0, 10))[0]


def test_direct_transposition_dominates_a_long_palindromic_walk() -> None:
    candidate = ((0, 3),)
    baseline = ((0, 1), (1, 2), (2, 3), (1, 2), (0, 1))
    assert sabre._restore_no_later(candidate, baseline, 4)
    for arrivals in product((0, 1, 100), repeat=4):
        assert all(
            new <= old
            for new, old in zip(
                _finish(candidate, arrivals), _finish(baseline, arrivals), strict=True
            )
        )


def test_all_five_wire_ring_permutations_restore_and_keep_tree_readiness() -> None:
    graph = CouplingMap.ring(5)
    # The existing BFS tree is 0--1--2 and 0--4--3.
    tree = [[1, 4], [0, 2], [1], [4], [0, 3]]
    improved = 0
    for layout in permutations(range(5)):
        baseline = sabre._walk_restore(layout, tree, tree)
        actual = sabre.restore_swaps(layout, graph)
        assert len(actual) <= len(baseline)
        assert sabre._restore_no_later(actual, baseline, 5)
        tokens = [layout.index(physical) for physical in range(5)]
        for left, right in actual:
            assert graph.has_edge(left, right)
            tokens[left], tokens[right] = tokens[right], tokens[left]
        assert tokens == list(range(5))
        improved += len(actual) < len(baseline)
    assert improved > 0


def test_adjacent_ring_token_exchange_still_uses_direct_device_edge() -> None:
    assert sabre.restore_swaps((0, 1, 2, 4, 3, 5), CouplingMap.ring(6)) == ((3, 4),)


def test_over_budget_readiness_proof_keeps_the_existing_restore() -> None:
    graph = CouplingMap.ring(6)
    layout = (0, 1, 2, 4, 3, 5)
    with patch.object(sabre, "_RESTORE_READINESS_OPERATION_BUDGET", 0):
        retained = sabre.restore_swaps(layout, graph)
    assert len(retained) == 9
    tokens = list(layout)
    for left, right in retained:
        tokens[left], tokens[right] = tokens[right], tokens[left]
    assert tokens == list(range(6))
