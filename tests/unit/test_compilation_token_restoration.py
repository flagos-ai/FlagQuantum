"""Check complete token restoration against independent permutation search."""

from collections import deque
from itertools import permutations

import pytest

from flagquantum.compiler.topology._token_restoration import _token_swaps
from flagquantum.compiler.topology.coupling import CouplingMap

pytestmark = pytest.mark.unit


def _minimum_swaps(graph: CouplingMap) -> dict[tuple[int, ...], int]:
    identity = tuple(range(graph.n_qubits))
    distances = {identity: 0}
    queue = deque([identity])
    while queue:
        placement = queue.popleft()
        for left, right in graph.edges:
            moved = tuple(
                right if wire == left else left if wire == right else wire
                for wire in placement
            )
            if moved not in distances:
                distances[moved] = distances[placement] + 1
                queue.append(moved)
    return distances


@pytest.mark.parametrize(
    "graph",
    [
        CouplingMap.line(5),
        CouplingMap.ring(5),
        CouplingMap(5, ((0, 1), (0, 2), (0, 3), (0, 4))),
        CouplingMap(5, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3))),
        CouplingMap(5, ((i, j) for i in range(5) for j in range(i + 1, 5))),
    ],
)
def test_every_five_wire_permutation_restores(graph: CouplingMap) -> None:
    minimum = _minimum_swaps(graph)
    for layout in permutations(range(5)):
        moves = _token_swaps(layout, graph, 4 * 5**2)
        assert moves is not None
        assert minimum[layout] <= len(moves) <= 4 * minimum[layout]
        actual = list(layout)
        for left, right in moves:
            assert (left, right) in graph.edges
            actual = [
                right if wire == left else left if wire == right else wire
                for wire in actual
            ]
        assert actual == list(range(5))
        assert _token_swaps(layout, graph, len(moves)) == moves
        if moves:
            assert _token_swaps(layout, graph, len(moves) - 1) is None


def test_unreachable_components_abstain() -> None:
    graph = CouplingMap(4, ((0, 1), (2, 3)))
    assert _token_swaps((2, 1, 0, 3), graph, 10) is None
    assert _token_swaps((1, 0, 3, 2), graph, 10) == ((0, 1), (2, 3))


def test_unused_device_wires_cannot_supply_shortcuts() -> None:
    graph = CouplingMap(4, ((0, 3), (1, 3), (2, 3)))
    assert _token_swaps((1, 0, 2), graph, 10) is None
    assert _token_swaps((0, 1, 2), graph, 0) == ()
