"""Small independent optimization oracles and pinned SDK conformance."""

from __future__ import annotations

import itertools
from typing import Literal, NoReturn

import pytest
import torch

pytest.importorskip("pymatching")
import pymatching

from flagquantum.errors import CapabilityError
from flagquantum.qec.adapters.pymatching_graph import PyMatchingGraphDecoder
from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph

pytestmark = pytest.mark.integration


def _optimum(graph: MatchingGraph, events: tuple[int, ...]) -> float:
    candidates = []
    for selected in itertools.product((False, True), repeat=len(graph.edges)):
        parity: set[int] = set()
        weight = 0.0
        for take, edge in zip(selected, graph.edges, strict=True):
            if take:
                parity.symmetric_difference_update(edge.detectors)
                weight += edge.weight
        parity.difference_update(graph.boundary_nodes)
        if parity == set(events):
            candidates.append(weight)
    return min(candidates)


@pytest.mark.parametrize(
    "costs", [(2.0, 3.0, 1.0), (-2.0, 3.0, 1.0), (-2.0, -3.0, -1.0), (0.0, 0.0, 0.0)]
)
def test_every_small_syndrome_has_an_independent_optimum(
    costs: tuple[float, float, float],
) -> None:
    graph = MatchingGraph(
        2,
        2,
        (
            MatchingEdge((0,), costs[0], (0,)),
            MatchingEdge((1,), costs[1], (1,)),
            MatchingEdge((0, 1), costs[2]),
        ),
    )
    decoder = PyMatchingGraphDecoder(graph)
    for pattern in itertools.product((False, True), repeat=2):
        events = tuple(i for i, active in enumerate(pattern) if active)
        solution = decoder.decode(events)
        assert solution.weight == _optimum(graph, events)
        assert solution.weight == sum(
            graph.edges[i].weight for i in solution.edge_indices
        )


@pytest.mark.parametrize(
    "packed_in,packed_out", [(False, False), (False, True), (True, False), (True, True)]
)
def test_matrix_input_batch_and_upstream(packed_in: bool, packed_out: bool) -> None:
    checks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
    labels = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
    graph, columns = MatchingGraph.from_check_matrix(
        checks,
        faults_matrix=labels,
        weights=torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64),
    )
    assert columns == (0, 1, 2)
    decoder = PyMatchingGraphDecoder(graph)
    dense = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.uint8)
    events = (
        torch.tensor([[0], [1], [2], [3]], dtype=torch.uint8) if packed_in else dense
    )
    bits, costs = decoder.decode_batch(
        events, bit_packed_shots=packed_in, bit_packed_predictions=packed_out
    )
    ref = pymatching.Matching.from_check_matrix(
        checks.numpy(),
        weights=[2.0, 3.0, 1.0],
        faults_matrix=labels.numpy(),
        merge_strategy="disallow",
        use_virtual_boundary_node=True,
    )
    expected, weights = ref.decode_batch(
        events.numpy(),
        return_weights=True,
        bit_packed_shots=packed_in,
        bit_packed_predictions=packed_out,
    )
    assert torch.equal(bits, torch.as_tensor(expected).to(bits.dtype))
    assert torch.equal(costs, torch.as_tensor(weights))
    weights.fill(-99)
    assert bool((costs >= 0).all())


def test_empty_syndrome_keeps_negative_cycle_and_endpoint_restriction() -> None:
    graph = MatchingGraph(
        3,
        0,
        (
            MatchingEdge((0, 1), -1.0),
            MatchingEdge((1, 2), -1.0),
            MatchingEdge((0, 2), -1.0),
        ),
    )
    decoder = PyMatchingGraphDecoder(graph)
    assert decoder.decode(()).edge_indices == (0, 1, 2)
    assert decoder.decode(()).weight == -3.0
    bits, costs = decoder.decode_batch(torch.zeros((1, 3), dtype=torch.uint8))
    assert bits.shape == (1, 0) and costs.tolist() == [-3.0]
    with pytest.raises(CapabilityError):
        decoder.decode_to_matched_detectors(())


def test_isolated_nodes_and_explicit_boundaries() -> None:
    graph = MatchingGraph(4, 4, (MatchingEdge((0, 3), 1.0, (2,)),), (3,))
    decoder = PyMatchingGraphDecoder(graph)
    assert decoder.decode((0,)).fault_ids == (2,)
    assert decoder.decode_to_matched_detectors((0,)) == ((0, None),)
    with pytest.raises(CapabilityError):
        decoder.decode((1,))


@pytest.mark.parametrize(
    "policy", ["independent", "smallest-weight", "keep-original", "replace"]
)
def test_merge_policy_matches_upstream(
    policy: Literal["independent", "smallest-weight", "keep-original", "replace"],
) -> None:
    checks = torch.ones((1, 2), dtype=torch.uint8)
    costs = torch.tensor([2.0, 1.0], dtype=torch.float64)
    rates = torch.tensor([0.2, 0.3], dtype=torch.float64)
    graph, columns = MatchingGraph.from_check_matrix(
        checks, weights=costs, error_probabilities=rates, merge_strategy=policy
    )
    reference = pymatching.Matching.from_check_matrix(
        checks.numpy(),
        weights=costs.numpy(),
        error_probabilities=rates.numpy(),
        merge_strategy=policy,
        use_virtual_boundary_node=True,
    )
    assert columns == (0, 0)
    attributes = reference.edges()[0][2]
    assert graph.edges[0].fault_ids == tuple(sorted(attributes["fault_ids"]))
    assert graph.edges[0].weight == pytest.approx(
        attributes["weight"], abs=1e-12, rel=0
    )
    assert graph.edges[0].error_probability == pytest.approx(
        attributes["error_probability"], abs=1e-12, rel=0
    )


@pytest.mark.parametrize("virtual", [False, True])
def test_repeated_matrix_boundary_geometry_matches_upstream(virtual: bool) -> None:
    checks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
    costs = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64)
    times = torch.tensor([4.0, 5.0], dtype=torch.float64)
    graph, columns = MatchingGraph.from_check_matrix(
        checks,
        weights=costs,
        repetitions=2,
        timelike_weights=times,
        use_virtual_boundary_node=virtual,
    )
    reference = pymatching.Matching.from_check_matrix(
        checks.numpy(),
        weights=costs.numpy(),
        repetitions=2,
        timelike_weights=times.numpy(),
        merge_strategy="disallow",
        use_virtual_boundary_node=virtual,
    )
    assert columns == (0, 1, 2)
    assert graph.num_nodes == reference.num_nodes
    assert set(graph.boundary_nodes) == reference.boundary
    expected = {}
    for first, second, data in reference.edges():
        nodes = (
            (int(first),)
            if second is None
            else tuple(sorted((int(first), int(second))))
        )
        expected[nodes] = (float(data["weight"]), tuple(sorted(data["fault_ids"])))
    assert {
        edge.detectors: (edge.weight, edge.fault_ids) for edge in graph.edges
    } == expected


@pytest.mark.parametrize("consumer", ["probability", "weighted_graph"])
def test_dependency_loader_replacement_reaches_both_consumers(
    consumer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder
    from flagquantum.qec.adapters import pymatching as owner

    error = owner.MatchingDependencyError("controlled missing matching dependency")

    def missing() -> NoReturn:
        raise error

    monkeypatch.setattr(owner, "_pymatching", missing)
    with pytest.raises(owner.MatchingDependencyError) as caught:
        if consumer == "probability":
            PyMatchingDecoder.from_detector_error_model(
                DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
            )
        else:
            PyMatchingGraphDecoder(
                MatchingGraph(1, 1, (MatchingEdge((0,), 1.0, (0,)),))
            )
    assert caught.value is error
