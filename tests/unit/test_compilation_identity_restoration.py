"""Check restored physical ownership for idle and active layouts."""

from __future__ import annotations

import itertools

import pytest

from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.compiler.topology.sabre import restore_swaps

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("edges", [(), ((0, 1), (2, 3)), ((0, 1), (1, 2), (2, 3))])
def test_identity_layout_needs_no_swaps_even_across_components(
    edges: tuple[tuple[int, int], ...],
) -> None:
    assert restore_swaps((0, 1, 2, 3), CouplingMap(4, edges)) == ()


@pytest.mark.parametrize("layout", [(0, 0), (0, 2), (-1, 0)])
def test_invalid_layouts_still_refuse_before_restoration(
    layout: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="permutation"):
        restore_swaps(layout, CouplingMap.line(2))


@pytest.mark.parametrize("topology", ["line", "ring"])
def test_every_five_wire_permutation_restores_physical_ownership(topology: str) -> None:
    coupling = CouplingMap.line(5) if topology == "line" else CouplingMap.ring(5)
    for layout in itertools.permutations(range(5)):
        occupants = [0] * 5
        for logical, physical in enumerate(layout):
            occupants[physical] = logical
        for left, right in restore_swaps(layout, coupling):
            assert coupling.has_edge(left, right)
            occupants[left], occupants[right] = occupants[right], occupants[left]
        assert occupants == list(range(5))
