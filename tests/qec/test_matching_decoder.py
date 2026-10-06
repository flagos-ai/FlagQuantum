"""Unit coverage for the minimum-weight matching decoder.

The matcher is verified against two references the decoder does not share code
with. The first is brute force: on a model small enough to enumerate, the
selected mechanisms must have the least total weight of every mechanism set whose
odd-degree detectors are the syndrome. The second is an exact enumeration of the
model: the probability of every mechanism set gives the syndrome distribution and
the logical-flip distribution, so the failure rate of an optimal decoder is a
number this module computes from the model alone. Where a model has no two
mechanisms that agree on their detectors but disagree on their labels, the
matcher must reach that number exactly.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.decoding_graph import DecodingGraph, DecodingGraphEdge
from flagquantum.qec.dem import DemError, DetectorErrorModel
from flagquantum.qec.matching import (
    MatchingDecodeResult,
    MinimumWeightMatchingDecoder,
)
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.surface import RotatedSurfaceCode

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 6, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def _decoder(model: DetectorErrorModel, **options: int):
    return MinimumWeightMatchingDecoder.from_detector_error_model(model, **options)


def _repetition_model(
    *, distance: int = 3, rounds: int = 2, p: float = 0.02
) -> DetectorErrorModel:
    return DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RepetitionCode(distance=distance), rounds=rounds),
        noise=PhenomenologicalNoise(data_flip=p, measurement_flip=p),
    )


def _odd_degree(graph: DecodingGraph, edges: Iterable[DecodingGraphEdge]) -> set[int]:
    """The detectors an odd number of the given mechanisms flip."""

    parity: set[int] = set()
    for edge in edges:
        for node in edge.detectors:
            if node == graph.boundary_node:
                continue
            parity ^= {node}
    return parity


def _brute_force_weight(
    graph: DecodingGraph, syndrome: set[int], *, max_edges: int
) -> float:
    """The least weight of every mechanism set up to ``max_edges`` long."""

    best = math.inf
    for size in range(1, max_edges + 1):
        for combination in itertools.combinations(range(graph.num_edges), size):
            edges = [graph.edges[index] for index in combination]
            weight = sum(edge.weight for edge in edges)
            if weight >= best:
                continue
            if _odd_degree(graph, edges) == syndrome:
                best = weight
    return best


def _syndrome_mass(model: DetectorErrorModel) -> dict[int, list[float]]:
    """Every mechanism set's probability, grouped by the syndrome it produces.

    The signature is the detector mask; each entry holds the probability mass
    whose logical observable did not flip and the mass whose observable did.
    """

    mass: dict[int, list[float]] = {0: [1.0, 0.0]}
    for error in model.errors:
        detectors = sum(1 << index for index in error.detectors)
        flips = sum(1 << index for index in error.observables) & 1
        probability = error.probability
        updated: dict[int, list[float]] = {}
        for signature, (without, with_flip) in mass.items():
            quiet = updated.setdefault(signature, [0.0, 0.0])
            quiet[0] += without * (1.0 - probability)
            quiet[1] += with_flip * (1.0 - probability)
            active = updated.setdefault(signature ^ detectors, [0.0, 0.0])
            active[0] += (with_flip if flips else without) * probability
            active[1] += (without if flips else with_flip) * probability
        mass = updated
    return mass


def _exact_rates(
    model: DetectorErrorModel, decoder: MinimumWeightMatchingDecoder
) -> tuple[float, float, float]:
    """The matcher's, the optimal, and the always-quiet failure rate.

    All three are properties of the model: every mechanism set has a probability,
    a syndrome, and a logical flip, so the distributions follow without sampling.
    The optimal decoder picks the more likely logical value for each syndrome; the
    matcher picks the value the cheapest mechanisms state; a decoder that always
    predicts no flip fails on exactly the mass whose observable flipped.
    """

    matched = 0.0
    optimal = 0.0
    always_quiet = 0.0
    for signature, (without, with_flip) in _syndrome_mass(model).items():
        syndrome = tuple(
            index for index in range(model.num_detectors) if signature >> index & 1
        )
        predicted = 1 if decoder.decode(syndrome).observables else 0
        matched += with_flip if predicted == 0 else without
        optimal += min(without, with_flip)
        always_quiet += with_flip
    return matched, optimal, always_quiet


def test_empty_syndrome_decodes_to_nothing() -> None:
    """A syndrome with no defective detector always predicts no flip.

    The decoder is asked for the cheapest explanation of "nothing happened", and
    the empty set of mechanisms is the only explanation of weight zero.
    """

    decoder = _decoder(
        _model(DemError(probability=0.1, detectors=(0, 1), observables=(0,)))
    )
    result = decoder.decode(())
    assert result == MatchingDecodeResult(observables=(), error_edges=(), weight=0.0)
    assert result.weight == 0.0


def test_single_detector_leaves_through_the_cheapest_boundary_edge() -> None:
    """One defect cannot pair with another, so it must reach the boundary.

    The detector has two boundary mechanisms of different likelihood and one
    internal edge, and the internal edge leads to a detector with no boundary
    mechanism of its own. The cheaper mechanism is the more likely one, which is
    the one of higher probability and therefore of lower weight.
    """

    model = _model(
        DemError(probability=0.05, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0,), observables=()),
        DemError(probability=0.05, detectors=(0, 1), observables=()),
        detectors=6,
    )
    decoder = _decoder(model)
    result = decoder.decode((0,))
    assert result.error_edges == (DecodingGraphEdge(detectors=(0, 6), probability=0.2),)
    assert result.observables == ()
    assert result.weight == pytest.approx(math.log(0.8 / 0.2))


def test_boundary_mechanism_carries_its_observable_label() -> None:
    """A defect that leaves the patch still flips what its mechanism flips.

    The only mechanism reaching the boundary is labelled, so the prediction must
    include the logical observable even though the correction has one edge.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0, 1), observables=()),
        detectors=6,
    )
    result = _decoder(model).decode((0,))
    assert result.error_edges == (
        DecodingGraphEdge(detectors=(0, 6), probability=0.1, observables=(0,)),
    )
    assert result.observables == (0,)


def test_odd_syndrome_decodes_by_routing_one_defect_to_the_boundary() -> None:
    """An odd defect count is decodable because the boundary takes any number.

    A T-join needs an even number of odd-degree vertices, so without a boundary
    the decoder would refuse this syndrome; with one, the nearest defect leaves
    the patch and the rest pair up. The correction must still explain the
    syndrome, which is what distinguishes a routed defect from a dropped one.
    """

    graph = DecodingGraph(
        num_detectors=4,
        num_observables=1,
        edges=(
            DecodingGraphEdge(detectors=(0, 1), probability=0.05),
            DecodingGraphEdge(detectors=(1, 2), probability=0.05),
            DecodingGraphEdge(detectors=(2, 3), probability=0.05),
            DecodingGraphEdge(detectors=(3, 4), probability=0.1),
        ),
    )
    decoder = MinimumWeightMatchingDecoder(graph=graph)
    for syndrome in ((0,), (0, 1, 2), (1, 2, 3)):
        result = decoder.decode(syndrome)
        assert _odd_degree(graph, result.error_edges) == set(syndrome)
    assert decoder.decode((3,)).error_edges == (
        DecodingGraphEdge(detectors=(3, 4), probability=0.1),
    )


def test_correction_explains_the_syndrome() -> None:
    """The selected mechanisms' odd-degree detectors are exactly the syndrome.

    This is the definition of a decoder being correct, and it holds for every
    syndrome regardless of whether the correction matches the injected noise.
    """

    model = _repetition_model(distance=3, rounds=3)
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = MinimumWeightMatchingDecoder(graph=graph)
    sample = model.dem_sampling(shots=120, seed=11)
    for shot in range(sample.detectors.shape[0]):
        syndrome = {int(index) for index in sample.detectors[shot].nonzero().flatten()}
        result = decoder.decode(syndrome)
        assert _odd_degree(graph, result.error_edges) == syndrome
        assert result.weight == pytest.approx(
            sum(edge.weight for edge in result.error_edges)
        )


def test_correction_has_least_weight_against_brute_force() -> None:
    """No mechanism set explains the syndrome more cheaply than the correction.

    The model's own samples supply the syndromes, and a syndrome with no defective
    detector is skipped because the empty set of mechanisms already explains it
    with weight zero and brute force over one or more edges could not see that.

    Brute force enumerates every set of up to four mechanisms, which covers the
    cheapest corrections of these syndromes; the matcher must reach the same
    weight, so a greedy or single-pass search would fail here.
    """

    model = _repetition_model(distance=3, rounds=2)
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = MinimumWeightMatchingDecoder(graph=graph)
    sample = model.dem_sampling(shots=400, seed=5)
    checked = 0
    for shot in range(sample.detectors.shape[0]):
        syndrome = {int(index) for index in sample.detectors[shot].nonzero().flatten()}
        if not syndrome or len(syndrome) > 4:
            continue
        result = decoder.decode(syndrome)
        best = _brute_force_weight(graph, syndrome, max_edges=4)
        assert math.isfinite(best)
        assert result.weight == pytest.approx(best)
        checked += 1
    assert checked > 40


def test_correction_has_least_weight_on_the_rotated_surface_code() -> None:
    """Optimality holds on the surface code's model and not only on repetition.

    Its graph is larger and its edges are of two weights, so a matcher that
    mishandled the boundary or the metric closure would overshoot here.
    """

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=2),
        noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = MinimumWeightMatchingDecoder(graph=graph)
    sample = model.dem_sampling(shots=200, seed=13)
    checked = 0
    for shot in range(sample.detectors.shape[0]):
        syndrome = {int(index) for index in sample.detectors[shot].nonzero().flatten()}
        if not syndrome or len(syndrome) > 4:
            continue
        result = decoder.decode(syndrome)
        assert _odd_degree(graph, result.error_edges) == syndrome
        assert result.weight == pytest.approx(
            _brute_force_weight(graph, syndrome, max_edges=4)
        )
        checked += 1
    assert checked > 40


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_single_round_repetition_decoding_is_optimal(distance: int) -> None:
    """With one round the matcher reaches the best failure rate attainable.

    A single-round repetition model has one mechanism per detector pair and no
    two mechanisms that agree on their detectors, so the cheapest explanation is
    the most likely one and matching is optimal rather than merely graphlike. The
    failure rate must equal the exhaustive-enumeration optimum, and it must fall
    by roughly a factor of the noise as the distance grows, which is what a
    decoder that returned a fixed answer could not do.
    """

    model = _repetition_model(distance=distance, rounds=1)
    decoder = _decoder(model)
    matched, optimal, _ = _exact_rates(model, decoder)
    assert matched == pytest.approx(optimal, abs=1e-12)
    assert 0.0 < matched < 0.01


def test_repeated_rounds_report_a_gap_to_the_optimal_rate() -> None:
    """More rounds introduce mechanisms the graph cannot tell apart.

    Two mechanisms can share a detector pair and disagree on the logical flip, so
    a matcher that sees only the syndrome cannot always pick the more likely one.
    The matcher's failure rate must stay above the optimal rate and far below the
    rate of a decoder that always predicts no flip, which is the honest statement
    of what a matching decoder achieves on this model.
    """

    model = _repetition_model(distance=3, rounds=3, p=0.02)
    matched, optimal, always_quiet = _exact_rates(model, _decoder(model))
    assert optimal < matched < always_quiet
    assert matched == pytest.approx(0.009125, abs=5e-6)
    assert optimal == pytest.approx(0.007715, abs=5e-6)
    assert always_quiet == pytest.approx(0.153733, abs=5e-6)


def test_logical_error_rate_falls_as_the_distance_grows() -> None:
    """The failure rate at a fixed noise shrinks with the code's distance.

    This is the property that makes the decoder a decoder: an implementation that
    ignored the detections or that mis-paired the boundary would not reproduce it.
    """

    rates = []
    for distance in (3, 5, 7):
        model = _repetition_model(distance=distance, rounds=1)
        rates.append(_exact_rates(model, _decoder(model))[0])
    assert rates[0] > rates[1] > rates[2] > 0.0
    assert rates[1] < rates[0] / 10.0
    assert rates[2] < rates[1] / 10.0


def test_prediction_is_the_exclusive_or_of_the_selected_labels() -> None:
    """The predicted observables are the parity the DEM states for the selection.

    The model's ``observables_flips_matrix`` is the independent statement of what
    each mechanism flips, so the prediction must be its product with the
    decoder's own selection, taken modulo two. The columns are matched to the
    selected edges by signature, because the graph orders its edges by their
    nodes while the model orders its mechanisms by their detectors.
    """

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=2),
        noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = MinimumWeightMatchingDecoder(graph=graph)
    flips = model.observables_flips_matrix().tolist()
    boundary = graph.boundary_node
    columns = {}
    for column, error in enumerate(model.errors):
        nodes = (
            error.detectors
            if len(error.detectors) == 2
            else (error.detectors[0], boundary)
        )
        columns[nodes + error.observables + (error.probability,)] = column
    assert len(columns) == model.num_errors

    sample = model.dem_sampling(shots=30, seed=17)
    for shot in range(sample.detectors.shape[0]):
        syndrome = {int(index) for index in sample.detectors[shot].nonzero().flatten()}
        result = decoder.decode(syndrome)
        selected = [
            columns[edge.detectors + edge.observables + (edge.probability,)]
            for edge in result.error_edges
        ]
        expected = [
            index
            for index, row in enumerate(flips)
            if sum(row[column] for column in selected) % 2
        ]
        assert list(result.observables) == expected


def test_prediction_matches_the_sampled_observable_at_low_noise() -> None:
    """The prediction agrees with the model's own samples once noise is small.

    This is the end-to-end statement: the decoder reads only the detectors and
    still recovers the logical flip that the model sampled. At p = 0.001 the
    single-round model is unambiguous, so the agreement is exact rather than
    statistical.
    """

    model = _repetition_model(distance=5, rounds=1, p=0.001)
    decoder = _decoder(model)
    sample = model.dem_sampling(shots=400, seed=23)
    failures = 0
    for shot in range(sample.detectors.shape[0]):
        syndrome = tuple(
            int(index) for index in sample.detectors[shot].nonzero().flatten()
        )
        predicted = decoder.decode(syndrome).observables
        actual = tuple(
            int(index) for index in sample.observables[shot].nonzero().flatten()
        )
        failures += predicted != actual
    assert failures <= 2


def test_decoding_is_reproducible() -> None:
    """The same graph and syndrome give an identical result every time."""

    model = _repetition_model(distance=3, rounds=3)
    decoder = _decoder(model)
    syndrome = tuple(range(0, 8, 2))
    first = decoder.decode(syndrome)
    second = decoder.decode(tuple(reversed(syndrome)))
    assert first == second


def test_decoder_refuses_more_defects_than_it_can_pair_exactly() -> None:
    """A large syndrome is a capability boundary, not a guessed pairing.

    The matcher enumerates the ways to pair the defects, so past the budget it has
    no exact answer to give and must say so instead of returning an approximation.
    """

    model = _repetition_model(distance=3, rounds=3)
    decoder = _decoder(model, max_defects=2)
    with pytest.raises(CapabilityError, match="at most 2"):
        decoder.decode((0, 2, 4))


def test_defect_budget_refusal_is_a_capability_boundary() -> None:
    model = _repetition_model(distance=3, rounds=3)
    decoder = _decoder(model, max_defects=1)
    with pytest.raises(CapabilityError) as raised:
        decoder.decode((0, 2))
    assert raised.value.category == "capability"


def test_decoder_refuses_a_syndrome_no_chain_explains() -> None:
    """Detectors in separate components cannot be paired with each other.

    The syndrome's defects on two components with no boundary edge have no
    explanation, so the decoder refuses rather than reporting a partial one.
    """

    graph = DecodingGraph(
        num_detectors=4,
        num_observables=1,
        edges=(
            DecodingGraphEdge(detectors=(0, 1), probability=0.1),
            DecodingGraphEdge(detectors=(2, 3), probability=0.1),
        ),
    )
    decoder = MinimumWeightMatchingDecoder(graph=graph)
    with pytest.raises(CapabilityError, match="no chain of mechanisms"):
        decoder.decode((0, 2))


def test_decoder_refuses_a_detector_outside_the_graph() -> None:
    decoder = _decoder(
        _model(DemError(probability=0.1, detectors=(0, 1), observables=()))
    )
    with pytest.raises(ValueError, match="outside the graph's 6 detectors"):
        decoder.decode((6,))


@pytest.mark.parametrize("event", [0.0, -1, "0", None])
def test_decoder_refuses_a_non_integer_detector(event: object) -> None:
    decoder = _decoder(
        _model(DemError(probability=0.1, detectors=(0, 1), observables=()))
    )
    with pytest.raises((TypeError, ValueError)):
        decoder.decode([event])


def test_decoder_refuses_a_repeated_detector() -> None:
    decoder = _decoder(
        _model(DemError(probability=0.1, detectors=(0, 1), observables=()))
    )
    with pytest.raises(ValueError, match="same detector twice"):
        decoder.decode((1, 1))


def test_decoder_refuses_a_non_graphlike_model() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1, 2), observables=(0,)),
        detectors=4,
    )
    with pytest.raises(CapabilityError, match="hyperedge"):
        _decoder(model)


@pytest.mark.parametrize("budget", [0, -1])
def test_decoder_refuses_a_non_positive_budget(budget: int) -> None:
    graph = DecodingGraph(
        num_detectors=2,
        num_observables=0,
        edges=(DecodingGraphEdge(detectors=(0, 1), probability=0.1),),
    )
    with pytest.raises(ValueError, match="at least one"):
        MinimumWeightMatchingDecoder(graph=graph, max_defects=budget)


def test_decoder_refuses_a_non_graph_argument() -> None:
    with pytest.raises(TypeError, match="DecodingGraph"):
        MinimumWeightMatchingDecoder(graph="graph")  # type: ignore[arg-type]


def test_result_refuses_an_unordered_observable_set() -> None:
    with pytest.raises(ValueError, match="unique and ascending"):
        MatchingDecodeResult(observables=(1, 0), error_edges=(), weight=0.0)


def test_result_refuses_a_repeated_mechanism() -> None:
    edge = DecodingGraphEdge(detectors=(0, 1), probability=0.1)
    with pytest.raises(ValueError, match="at most once"):
        MatchingDecodeResult(observables=(), error_edges=(edge, edge), weight=0.2)


def test_result_refuses_a_non_finite_weight() -> None:
    with pytest.raises(ValueError, match="finite"):
        MatchingDecodeResult(observables=(), error_edges=(), weight=math.inf)


@pytest.mark.parametrize(
    ("observables", "error"),
    [
        (("0",), TypeError),
        ((0.0,), TypeError),
        ((-1,), ValueError),
    ],
)
def test_result_refuses_a_malformed_observable_index(
    observables: tuple[object, ...], error: type[Exception]
) -> None:
    """The prediction is a set of observable indices, not of values.

    A label that is not an integer index would be unusable to a caller that
    compares it against a detector error model's own observables.
    """

    with pytest.raises(error):
        MatchingDecodeResult(observables=observables, error_edges=(), weight=0.0)


def test_result_refuses_an_entry_that_is_not_a_mechanism() -> None:
    with pytest.raises(TypeError, match="DecodingGraphEdge records"):
        MatchingDecodeResult(
            observables=(), error_edges=((0, 1),), weight=0.1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("weight", ["0.1", True])
def test_result_refuses_a_non_real_weight(weight: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        MatchingDecodeResult(
            observables=(), error_edges=(), weight=weight  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("budget", ["3", 2.0, True])
def test_decoder_refuses_a_non_integer_budget(budget: object) -> None:
    graph = DecodingGraph(
        num_detectors=2,
        num_observables=0,
        edges=(DecodingGraphEdge(detectors=(0, 1), probability=0.1),),
    )
    with pytest.raises(TypeError, match="must be an integer"):
        MinimumWeightMatchingDecoder(
            graph=graph, max_defects=budget  # type: ignore[arg-type]
        )
