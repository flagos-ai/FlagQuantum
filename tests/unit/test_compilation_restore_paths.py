"""Layout restoration preserves tokens without repeated path searches."""

from __future__ import annotations

from itertools import permutations
from unittest.mock import patch

import pytest

from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology.coupling import CouplingMap

pytestmark = pytest.mark.unit


def _assert_restored(
    layout: tuple[int, ...], graph: CouplingMap, swaps: tuple[tuple[int, int], ...]
) -> None:
    tokens = [0] * len(layout)
    for logical, physical in enumerate(layout):
        tokens[physical] = logical
    for left, right in swaps:
        assert graph.has_edge(left, right)
        assert left < len(layout) and right < len(layout)
        tokens[left], tokens[right] = tokens[right], tokens[left]
    assert tokens == list(range(len(layout)))


@pytest.mark.parametrize(
    "graph",
    (
        CouplingMap.line(5),
        CouplingMap(5, ((0, 1), (0, 2), (0, 3), (0, 4))),
        CouplingMap(5, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))),
    ),
)
def test_every_five_wire_permutation_restores_on_device_edges(
    graph: CouplingMap,
) -> None:
    for layout in permutations(range(5)):
        swaps = sabre.restore_swaps(layout, graph)
        _assert_restored(layout, graph, swaps)
        assert sabre.restore_swaps(layout, graph) == swaps


@pytest.mark.parametrize(
    ("layout", "graph", "expected"),
    (
        (
            (4, 3, 2, 1, 0),
            CouplingMap.line(5),
            (
                (3, 4),
                (2, 3),
                (1, 2),
                (0, 1),
                (3, 4),
                (2, 3),
                (1, 2),
                (3, 4),
                (2, 3),
                (3, 4),
            ),
        ),
        (
            (2, 4, 1, 0, 3),
            CouplingMap(5, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))),
            ((1, 2), (0, 4), (0, 1), (3, 4)),
        ),
        (
            (1, 0, 3, 2),
            CouplingMap(4, ((0, 1), (2, 3))),
            ((0, 1), (2, 3)),
        ),
    ),
)
def test_restoration_retains_existing_swap_order(
    layout: tuple[int, ...],
    graph: CouplingMap,
    expected: tuple[tuple[int, int], ...],
) -> None:
    assert sabre.restore_swaps(layout, graph) == expected
    _assert_restored(layout, graph, expected)


def test_long_line_walk_searches_once_per_displaced_leaf() -> None:
    width = 64
    layout = tuple(reversed(range(width)))
    graph = CouplingMap.line(width)
    with patch.object(sabre, "_restore_path", wraps=sabre._restore_path) as search:
        swaps = sabre.restore_swaps(layout, graph)
    _assert_restored(layout, graph, swaps)
    # Reversal requires this many adjacent transpositions, but only one path
    # search per displaced leaf. Timing is deliberately not a test assertion.
    assert len(swaps) == width * (width - 1) // 2
    assert search.call_count == width - 1


def test_identity_and_padding_need_no_extra_swaps() -> None:
    with patch.object(sabre, "_restore_path", wraps=sabre._restore_path) as search:
        assert sabre.restore_swaps((0, 1, 2), CouplingMap.line(5)) == ()
    assert search.call_count == 0
    layout = (2, 0, 1)
    _assert_restored(
        layout, CouplingMap.line(5), sabre.restore_swaps(layout, CouplingMap.line(5))
    )


def test_disconnected_component_exchange_remains_rejected() -> None:
    with pytest.raises(ValueError, match="no coupling path"):
        sabre.restore_swaps((2, 1, 0, 3), CouplingMap(4, ((0, 1), (2, 3))))


def test_invalid_permutation_remains_rejected() -> None:
    with pytest.raises(ValueError, match="permutation"):
        sabre.restore_swaps((0, 0, 2), CouplingMap.line(3))
