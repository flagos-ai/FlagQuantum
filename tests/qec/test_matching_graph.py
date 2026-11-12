"""Core matching contracts and a consumer-unchanged replacement fake."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Literal

import pytest
import torch

from flagquantum.errors import CapabilityError
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


def test_matrix_columns_costs_and_faults_stay_independent() -> None:
    checks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
    faults = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
    graph, columns = MatchingGraph.from_check_matrix(
        checks,
        faults_matrix=faults,
        weights=torch.tensor([2.0, 3.0, -1.0]),
        error_probabilities=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
    )
    assert columns == (0, 1, 2)
    assert tuple(edge.weight for edge in graph.edges) == (2.0, 3.0, -1.0)
    assert graph.edges[2].fault_ids == () and graph.edges[2].error_probability == 0.3
    exported, labels, weights = graph.to_check_matrices()
    assert torch.equal(exported, checks) and torch.equal(labels, faults)
    assert weights.tolist() == [2.0, 3.0, -1.0]
    checks.zero_()
    faults.zero_()
    assert graph.edges[2].detectors == (0, 1)


@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
def test_sparse_rounds_boundaries_and_unused_columns(layout: str) -> None:
    checks = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
    if layout == "coo":
        checks = checks.to_sparse().coalesce()
    elif layout == "csr":
        checks = checks.to_sparse_csr()
    graph, columns = MatchingGraph.from_check_matrix(
        checks, repetitions=2, use_virtual_boundary_node=False
    )
    assert columns == (0, 1, None)
    assert graph.num_nodes == 5 and graph.boundary_nodes == (4,)
    assert graph.num_fault_ids == 3
    assert [edge.detectors for edge in graph.edges] == [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (0, 2),
        (1, 3),
    ]


@pytest.mark.parametrize(
    "policy,cost,label",
    [
        ("smallest-weight", 1.0, (1,)),
        ("replace", 1.0, (1,)),
        ("keep-original", 2.0, (0,)),
    ],
)
def test_explicit_parallel_policy(
    policy: Literal["smallest-weight", "replace", "keep-original"],
    cost: float,
    label: tuple[int, ...],
) -> None:
    graph, columns = MatchingGraph.from_check_matrix(
        torch.ones((1, 2), dtype=torch.uint8),
        weights=torch.tensor([2.0, 1.0]),
        merge_strategy=policy,
    )
    assert (
        columns == (0, 0)
        and graph.edges[0].weight == cost
        and graph.edges[0].fault_ids == label
    )


def test_independent_merge_and_refusals() -> None:
    graph, columns = MatchingGraph.from_check_matrix(
        torch.ones((1, 2), dtype=torch.uint8),
        weights=0.0,
        error_probabilities=0.25,
        merge_strategy="independent",
    )
    assert columns == (0, 0) and graph.edges[0].weight == 0.0
    assert (
        graph.edges[0].fault_ids == (0,) and graph.edges[0].error_probability == 0.375
    )
    with pytest.raises(ValueError):
        MatchingGraph.from_check_matrix(torch.ones((1, 2), dtype=torch.uint8))
    with pytest.raises(CapabilityError):
        MatchingGraph.from_check_matrix(torch.ones((3, 1), dtype=torch.uint8))


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_binary_matrices_preserve_values(dtype: torch.dtype) -> None:
    graph, columns = MatchingGraph.from_check_matrix(
        torch.tensor([[1, 0], [0, 1]], dtype=dtype)
    )
    assert columns == (0, 1)
    assert [edge.detectors for edge in graph.edges] == [(0,), (1,)]
