"""Core matching contracts and a consumer-unchanged replacement fake."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from flagquantum.qec.decoders.matching_graph import (
    MatchingEdge,
    MatchingGraph,
    MatchingSolution,
)

pytestmark = pytest.mark.unit


def _unique_pair_consumer(
    graph: MatchingGraph, events: tuple[int, ...]
) -> MatchingSolution:
    """A contract fake for the unique one-edge problem; no SDK dependency."""
    assert graph == MatchingGraph(2, 1, (MatchingEdge((0, 1), -2.0, (0,), 0.125),))
    assert events == (0, 1)
    return MatchingSolution((0,), (0,), -2.0, -2.0)


def test_signed_cost_is_independent_of_probability() -> None:
    edge = MatchingEdge((1, 0), -2.0, (0,), 0.125)
    graph = MatchingGraph(2, 1, (edge,))
    result = _unique_pair_consumer(graph, (0, 1))
    assert result.weight == -2.0 and edge.error_probability == 0.125
    assert result.fault_ids == (0,) and result.edge_indices == (0,)
    with pytest.raises(FrozenInstanceError):
        setattr(edge, "weight", 1.0)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 2**24, -(2**24)])
def test_invalid_costs_fail_closed(weight: float) -> None:
    with pytest.raises(ValueError):
        MatchingEdge((0,), weight)


def test_shape_boundary_and_fault_identity() -> None:
    graph = MatchingGraph(4, 5, (MatchingEdge((0, 3), 0.0, (4,)),), (3,))
    assert graph.num_nodes == 4 and graph.num_fault_ids == 5
    assert graph.boundary_nodes == (3,) and graph.edges[0].fault_ids == (4,)
    with pytest.raises(ValueError):
        MatchingGraph(2, 1, (MatchingEdge((0, 1), 1.0, (1,)),))
    with pytest.raises(ValueError):
        MatchingGraph(2, 0, (MatchingEdge((0,), 1.0), MatchingEdge((0,), 2.0)))
