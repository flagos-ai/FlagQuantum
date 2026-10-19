"""Unit coverage for the decoding graph a detector error model defines.

The graph is verified against the model it came from rather than against a
transcribed edge list: every mechanism must become one edge whose weight is its
negative log-likelihood ratio and whose labels are exactly the observables the
model's own ``observables_flips_matrix`` states for it. That pins the graph
without depending on how this module orders its edges.
"""

from __future__ import annotations

import math

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.decoding_graph import DecodingGraph, DecodingGraphEdge
from flagquantum.qec.dem import DemError, DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.surface import RotatedSurfaceCode

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def _repetition_model(p: float = 0.02, rounds: int = 2) -> DetectorErrorModel:
    return DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RepetitionCode(distance=3), rounds=rounds),
        noise=PhenomenologicalNoise(data_flip=p, measurement_flip=p),
    )


def test_edge_weight_is_the_negative_log_likelihood_ratio() -> None:
    """The weight is ``log((1 - p) / p)`` and not ``-log(p)`` or ``p``.

    A matcher minimises the total weight, so the likelihood ratio is what makes
    the cheapest correction the most likely one. A weight of ``-log(p)`` would
    agree at a single edge and diverge as soon as two edges are compared.
    """

    edge = DecodingGraphEdge(detectors=(0, 1), probability=0.25)
    assert edge.weight == pytest.approx(math.log(3.0))
    assert edge.weight == pytest.approx(1.0986122886681098)
    assert DecodingGraphEdge(detectors=(0, 1), probability=0.5).weight == pytest.approx(
        0.0
    )


def test_edge_rejects_a_probability_above_one_half() -> None:
    """A mechanism more likely than not has no non-negative edge weight."""

    with pytest.raises(ValueError, match="no more likely than not"):
        DecodingGraphEdge(detectors=(0, 1), probability=0.6)


def test_edge_rejects_a_zero_probability() -> None:
    with pytest.raises(ValueError, match="positive probability"):
        DecodingGraphEdge(detectors=(0, 1), probability=0.0)


@pytest.mark.parametrize("nodes", [(0,), (0, 1, 2)])
def test_edge_rejects_a_node_count_other_than_two(nodes: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="exactly two nodes"):
        DecodingGraphEdge(detectors=nodes, probability=0.1)


def test_edge_rejects_a_non_integer_node() -> None:
    with pytest.raises(TypeError, match="nodes must be integers"):
        DecodingGraphEdge(detectors=(0, "1"), probability=0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("nodes", [(1, 0), (2, 2)])
def test_edge_rejects_nodes_that_are_not_ascending(nodes: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="ascending"):
        DecodingGraphEdge(detectors=nodes, probability=0.1)


def test_edge_normalizes_its_observables() -> None:
    edge = DecodingGraphEdge(detectors=(0, 1), probability=0.1, observables=(2, 0, 2))
    assert edge.observables == (0, 2)


def test_graph_derives_one_edge_per_mechanism() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.2, detectors=(2,), observables=()),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_detectors == 4
    assert graph.num_observables == 1
    assert graph.num_edges == 2
    assert graph.boundary_node == 4
    assert graph.edges[0] == DecodingGraphEdge(
        detectors=(0, 1), probability=0.1, observables=(0,)
    )
    # A single-detector mechanism becomes an edge to the boundary node, which is
    # one past the last detector.
    assert graph.edges[1] == DecodingGraphEdge(
        detectors=(2, 4), probability=0.2, observables=()
    )


def test_graph_drops_a_mechanism_no_syndrome_can_reveal() -> None:
    """An error that flips no detector changes nothing a decoder reads.

    The model itself refuses an empty signature, so this case can only arrive as
    an observable-only mechanism, which its ``__post_init__`` also refuses; the
    graph therefore never has to represent it. The assertion records that the
    graph carries one edge per mechanism rather than one per detector pair.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.1, detectors=(1,), observables=(0,)),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_edges == 2


def test_graph_keeps_parallel_edges_distinct() -> None:
    """Two mechanisms on the same detector pair with different labels stay two.

    Merging them into one edge would lose a mechanism that flips the logical
    observable, and averaging their probabilities would invent a weight neither
    mechanism has.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=()),
        DemError(probability=0.2, detectors=(0, 1), observables=(0,)),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_edges == 2
    assert [edge.observables for edge in graph.edges] == [(), (0,)]
    assert [edge.probability for edge in graph.edges] == [0.1, 0.2]


def test_graph_is_ordered_by_its_edges() -> None:
    """Two graphs of one model are equal, so the matcher is reproducible."""

    model = _model(
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
        DemError(probability=0.1, detectors=(0, 1), observables=()),
        DemError(probability=0.3, detectors=(0,), observables=(0,)),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert [edge.detectors for edge in graph.edges] == [(0, 1), (0, 4), (1, 4)]
    reversed_graph = DecodingGraph(
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        edges=tuple(reversed(graph.edges)),
    )
    assert reversed_graph == graph


def test_graph_refuses_a_hyperedge() -> None:
    """A mechanism flipping three detectors is not an edge, so it is refused.

    Projecting it onto a pair would silently drop one of its detectors and the
    matcher would then return a correction that does not explain the syndrome.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0, 1, 2), observables=(0,)),
        detectors=4,
    )
    with pytest.raises(CapabilityError, match="D0, D1, D2"):
        DecodingGraph.from_detector_error_model(model)


def test_graph_reports_the_mechanism_it_refuses_as_a_capability_boundary() -> None:
    """The refusal is a capability boundary and not a malformed-input error."""

    model = _model(
        DemError(probability=0.1, detectors=(0, 1, 3), observables=()),
        detectors=4,
    )
    with pytest.raises(CapabilityError) as raised:
        DecodingGraph.from_detector_error_model(model)
    assert raised.value.category == "capability"


def test_graph_refuses_an_edge_outside_its_detectors() -> None:
    with pytest.raises(ValueError, match="outside the graph"):
        DecodingGraph(
            num_detectors=2,
            num_observables=1,
            edges=(DecodingGraphEdge(detectors=(0, 3), probability=0.1),),
        )


def test_graph_refuses_an_edge_outside_its_observables() -> None:
    with pytest.raises(ValueError, match="observable index"):
        DecodingGraph(
            num_detectors=2,
            num_observables=1,
            edges=(
                DecodingGraphEdge(detectors=(0, 1), probability=0.1, observables=(1,)),
            ),
        )


def test_graph_refuses_an_entry_that_is_not_an_edge() -> None:
    with pytest.raises(TypeError, match="DecodingGraphEdge records"):
        DecodingGraph(
            num_detectors=2,
            num_observables=1,
            edges=((0, 1),),  # type: ignore[arg-type]
        )


def test_graph_requires_at_least_one_detector() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DecodingGraph(num_detectors=0, num_observables=0)


def test_graph_skips_a_mechanism_that_never_fires() -> None:
    """A mechanism of probability zero adds an edge no syndrome can select.

    The model keeps it because it is part of the model, but its weight would be
    infinite, so the graph would hand the matcher a path of unbounded cost.
    """

    model = _model(
        DemError(probability=0.0, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.1, detectors=(1,), observables=()),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_edges == 1
    assert graph.edges[0].probability == 0.1


def test_graph_refuses_a_non_model_argument() -> None:
    with pytest.raises(TypeError, match="DetectorErrorModel"):
        DecodingGraph.from_detector_error_model("error(0.1) D0\n")


def test_repetition_graph_matches_its_model_matrices() -> None:
    """Every edge agrees with the model on its detectors, labels, and weight.

    The model's matrices are the independent statement of the same facts: the
    edge's weight must be the log-likelihood ratio of the probability in the
    column it came from, and its labels must be that column's observable flips.
    Edges are keyed by their signature because the model and the graph order their
    entries differently: the model sorts a single-detector mechanism by its one
    detector while the graph sorts the same mechanism by its boundary edge.
    """

    model = _repetition_model()
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_detectors == model.num_detectors
    assert graph.num_observables == model.num_observables
    assert graph.num_edges == model.num_errors

    flips = model.observables_flips_matrix().tolist()
    boundary = graph.boundary_node
    edges = {
        edge.detectors + edge.observables + (edge.probability,): edge
        for edge in graph.edges
    }
    assert len(edges) == model.num_errors
    for column, error in enumerate(model.errors):
        expected = error.detectors
        nodes = expected if len(expected) == 2 else (expected[0], boundary)
        edge = edges[nodes + error.observables + (error.probability,)]
        assert edge.detectors == nodes
        assert edge.probability == error.probability
        assert edge.weight == pytest.approx(
            math.log((1 - error.probability) / error.probability)
        )
        assert edge.observables == tuple(
            index for index, row in enumerate(flips) if row[column]
        )


def test_repetition_graph_has_boundary_edges() -> None:
    """An odd syndrome is decodable only because the boundary is an edge target.

    The count comes from the model rather than from a transcription: one boundary
    edge per single-detector mechanism, which is what a data flip at the end of
    the repetition code's chain and the first round's measurement flips produce.
    """

    model = _repetition_model()
    graph = DecodingGraph.from_detector_error_model(model)
    boundary = graph.boundary_node
    assert boundary == model.num_detectors
    single = sum(1 for error in model.errors if len(error.detectors) == 1)
    assert single > 0
    assert sum(1 for edge in graph.edges if edge.detectors[1] == boundary) == single
    assert all(edge.detectors[0] < boundary for edge in graph.edges)


def test_surface_graph_is_graphlike_at_distance_three() -> None:
    """The rotated surface code's own model states no hyperedge at distance 3.

    This is the evidence for the graph's supported scope: the refusal above fires
    on a hand-built model, while the only code this package models at distance
    three stays inside the graph.
    """

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3),
        noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_detectors == 24
    assert graph.num_observables == 1
    assert graph.num_edges == model.num_errors == 45
    assert all(len(edge.detectors) == 2 for edge in graph.edges)
    assert all(edge.weight > 0.0 for edge in graph.edges)
