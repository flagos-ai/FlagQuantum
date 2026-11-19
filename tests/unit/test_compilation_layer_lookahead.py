"""Native layer proposals preserve dependencies and bounded search behavior."""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from unittest.mock import patch

import pytest

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology._layer_lookahead import choose_layer_swaps
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def test_disjoint_pair_resolves_single_swap_tie_without_mutating_inputs() -> None:
    pairs = ((0, 2),) * 8
    successors = [[i + 1] if i < 7 else [] for i in range(8)]
    pending = [0] + [1] * 7
    layout = [0, 2, 3, 1]
    arguments = (
        {(0, 1), (2, 3)},
        [0],
        pairs,
        (True,) * 8,
        successors,
        pending,
        layout,
        CouplingMap.line(4).distance_matrix(),
    )
    before = deepcopy(arguments)
    assert choose_layer_swaps(*arguments) == ((0, 1), (2, 3))
    assert arguments == before


def test_short_complete_horizon_retains_upstream_deterministic_tie() -> None:
    assert choose_layer_swaps(
        {(0, 1), (2, 3)},
        [0],
        ((0, 3),),
        (True,),
        [[]],
        [0],
        list(range(4)),
        CouplingMap.line(4).distance_matrix(),
    ) == ((0, 1),)


def test_dependency_join_waits_for_all_predecessors() -> None:
    # The omitted predecessor is not ready. Its dependent distant gate must not
    # influence the proposal before that dependency has executed.
    assert choose_layer_swaps(
        {(0, 1), (2, 3)},
        [0],
        ((0, 3), (1,), (1, 3)),
        (True, False, True),
        [[2], [2], []],
        [0, 1, 2],
        list(range(4)),
        CouplingMap.line(4).distance_matrix(),
    ) == ((0, 1),)


def test_projection_stops_before_reading_more_than_160_operations() -> None:
    class UnreadableSuccessors(list[list[int]]):
        def __getitem__(self, index: int) -> list[int]:
            raise AssertionError("oversized frontier must not be traversed")

    assert (
        choose_layer_swaps(
            {(0, 1)},
            list(range(161)),
            ((0, 3),) * 161,
            (True,) * 161,
            UnreadableSuccessors(),
            [0] * 161,
            list(range(4)),
            CouplingMap.line(4).distance_matrix(),
        )
        is None
    )


def test_pair_enumeration_is_skipped_above_32_candidates() -> None:
    pairs = ((0, 3),) * 8
    with patch(
        "flagquantum.compiler.topology._layer_lookahead.itertools.combinations",
        side_effect=AssertionError("pair budget exceeded"),
    ):
        assert (
            choose_layer_swaps(
                {(0, 1), (2, 3)} | {(q, q + 1) for q in range(4, 36)},
                [0],
                pairs,
                (True,) * 8,
                [[i + 1] if i < 7 else [] for i in range(8)],
                [0] + [1] * 7,
                list(range(37)),
                CouplingMap.line(37).distance_matrix(),
            )
            is None
        )


@pytest.mark.parametrize("seed", range(10))
def test_extra_candidate_never_increases_delivered_cost(seed: int) -> None:
    rng = random.Random(seed)
    source = CircuitIR(
        8, tuple(Instruction("cx", tuple(rng.sample(range(8), 2))) for _ in range(32))
    )
    coupling = CouplingMap.grid(2, 4)
    with patch.object(sabre, "choose_layer_swaps", return_value=None):
        previous = sabre.plan_sabre_swaps(source, coupling)
    selected = sabre.plan_sabre_swaps(source, coupling)
    assert sabre._restored_swap_count(selected, coupling) <= sabre._restored_swap_count(
        previous, coupling
    )


def test_selection_matches_cirq_source_on_small_dependency_graphs() -> None:
    pytest.importorskip("cirq")
    from cirq.transformers.routing.route_circuit_cqc import RouteCQC

    class Mapping:
        def __init__(self, layout: list[int], distances: tuple[tuple[float, ...], ...]):
            self.layout = list(layout)
            self.distances = distances

        def apply_swap(self, first: int, second: int) -> None:
            self.layout = [
                second if value == first else first if value == second else value
                for value in self.layout
            ]

        def dist_on_device(self, first: int, second: int, *, undirected: bool) -> float:
            assert undirected
            return self.distances[self.layout[first]][self.layout[second]]

    for width in (4, 6, 8):
        topology = CouplingMap.line(width)
        distances = topology.distance_matrix()
        for seed in range(100):
            rng = random.Random(seed)
            pairs = tuple(tuple(rng.sample(range(width), 2)) for _ in range(5 * width))
            successors: list[list[int]] = [[] for _ in pairs]
            pending = [0] * len(pairs)
            previous: dict[int, int] = {}
            for index, pair in enumerate(pairs):
                for qubit in pair:
                    if qubit in previous:
                        successors[previous[qubit]].append(index)
                        pending[index] += 1
                    previous[qubit] = index
            front = [i for i, degree in enumerate(pending) if degree == 0]
            remaining = pending.copy()
            ready = list(front)
            layers = []
            while ready:
                layers.append([pairs[i] for i in ready])
                following = []
                for index in ready:
                    for successor in successors[index]:
                        remaining[successor] -= 1
                        if remaining[successor] == 0:
                            following.append(successor)
                ready = following
            layout = list(range(width))
            rng.shuffle(layout)
            candidates = {
                tuple(sorted((layout[q], neighbor)))
                for index in front
                for q in pairs[index]
                for neighbor in topology.neighbors(layout[q])
            }
            manager = Mapping(layout, distances)
            expected = RouteCQC._choose_optimal_swap(
                manager, layers, 0, 8, [(edge,) for edge in sorted(candidates)]
            )
            if expected is None:
                options = [
                    (first, second)
                    for first, second in itertools.combinations(sorted(candidates), 2)
                    if set(first).isdisjoint(second)
                ]
                expected = (
                    RouteCQC._choose_optimal_swap(manager, layers, 0, 8, options)
                    if options
                    else None
                )
            actual = choose_layer_swaps(
                candidates,
                front,
                pairs,
                (True,) * len(pairs),
                successors,
                pending,
                layout,
                distances,
            )
            assert actual == expected, (width, seed)
            assert manager.layout == layout
