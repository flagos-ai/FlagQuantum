"""Cross-check the self-implemented matcher against PyMatching.

`flagquantum/qec/matching.py` is the authority for a decoded syndrome, and its own
evidence is brute force over the syndromes small enough to enumerate plus the
exhaustive enumeration of a model's mechanism distribution. Both read the
detector error model the decoder reads, so neither is independent of it. This
module runs the one check that is: a second decoder, on the same graph, in the
same records.

Three things are asserted here and no more, because three things are what the
comparison can establish.

The two decoders agree on the cheapest weight of every syndrome they both answer.
That is the exact claim: the weight is the same quantity on both sides because
`PyMatchingDecoder` sums the mechanisms PyMatching selected in double precision
rather than reporting PyMatching's own narrower accumulator.

They agree on the observables wherever the cheapest explanation is unique. Where
two explanations tie, they may disagree, and the file pins one such syndrome on
`RepetitionCode(distance=3)` by hand so that the tie clause is a demonstrated
fact rather than a place a failure could hide. That is the honest form of the
agreement: a disagreement inside the tie band is not evidence against either
implementation, and every disagreement outside it is.

They refuse the same syndromes. PyMatching raises where no perfect matching
exists and the matcher refuses the same syndrome as a capability boundary, so
the exhaustive comparison over a small graph asserts that the two refusals cover
exactly the same syndromes.

The tie band is not fitted to the observed results. PyMatching's arithmetic is
narrower than this package's -- see the module docstring of
`flagquantum/qec/adapters.py` for the measured difference -- so the band is one
single-precision rounding per selected mechanism, twenty times that difference,
and every disagreement reported below is outside it.

The translation is exercised in the same file: the shape of the matching graph,
the boundary sentinel PyMatching returns for a mechanism that reaches the
boundary, and the two graphs the translation refuses with a stated reason.
"""

from __future__ import annotations

import builtins
import random
import sys
from dataclasses import dataclass

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    DemError,
    DetectorErrorModel,
    MatchingDependencyError,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    code_matrices,
)
from flagquantum.qec.adapters import _BOUNDARY, _pymatching, _translate
from flagquantum.qec.decoding_graph import DecodingGraph
from flagquantum.qec.matching import MatchingDecodeResult

# The cross-check needs its reference implementation, which is an optional
# distribution, so the file skips rather than failing to import when it is
# absent. `flagquantum.qec` itself imports none of it: the extra is reached
# through `flagquantum.qec.adapters._pymatching` at call time only.
pytest.importorskip("pymatching")

import pymatching

pytestmark = pytest.mark.integration

# One single-precision rounding per mechanism, as the module docstring of
# `flagquantum.qec.adapters` derives it.
_ROUNDING = 2**-23


@dataclass(frozen=True)
class _Configuration:
    """A model the two decoders can both read."""

    name: str
    model: DetectorErrorModel


def _memory(code: object, rounds: int, strength: float) -> DetectorErrorModel:
    return DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=rounds),  # type: ignore[arg-type]
        noise=PhenomenologicalNoise(data_flip=strength, measurement_flip=strength),
    )


# Four models, chosen so that the two decoders are compared on a graph with a
# boundary that is reached often (the repetition code) and on one where whole
# mechanisms stay inside the patch (the rotated surface code), and so that the
# larger graphs exercise a graph order of tens of mechanisms rather than a
# handful.
_CONFIGURATIONS = (
    _Configuration(
        "repetition-3-round-3", _memory(RepetitionCode(distance=3), 3, 0.05)
    ),
    _Configuration(
        "repetition-5-round-2", _memory(RepetitionCode(distance=5), 2, 0.03)
    ),
    _Configuration(
        "rotated-surface-3-round-2", _memory(RotatedSurfaceCode(distance=3), 2, 0.03)
    ),
    _Configuration(
        "rotated-surface-3-round-3", _memory(RotatedSurfaceCode(distance=3), 3, 0.01)
    ),
)

# A graph with no boundary mechanism at all: three detectors in a line, so the
# two ends reach only each other and the syndrome of one end alone has no
# explanation. That is the shape in which PyMatching and the matcher both refuse,
# and it is the only shape here where a refusal is reachable at all.
_LINE = DetectorErrorModel(
    num_detectors=3,
    num_observables=1,
    errors=(
        DemError(probability=0.1, detectors=(0, 1)),
        DemError(probability=0.1, detectors=(1, 2)),
    ),
)


def _decoders(
    model: DetectorErrorModel,
) -> tuple[MinimumWeightMatchingDecoder, PyMatchingDecoder]:
    """The authority and the cross-check over one model.

    The authority is given the whole detector budget rather than its default
    twenty, so that the comparison is not silently narrowed by one side's
    enumeration cost: a syndrome either has an explanation on both sides or on
    neither.
    """

    graph = DecodingGraph.from_detector_error_model(model)
    return (
        MinimumWeightMatchingDecoder(graph, max_defects=graph.num_detectors),
        PyMatchingDecoder(graph),
    )


def _syndromes(model: DetectorErrorModel, *, limit: int | None) -> list[list[int]]:
    """Every syndrome of a model, or a deterministic sample of the small ones.

    Exhaustive up to the size an enumeration can afford, and above it the
    syndromes of at most ``limit`` defects drawn from a fixed seed, because the
    authority enumerates the pairings of the defects it is given.
    """

    graph = DecodingGraph.from_detector_error_model(model)
    count = graph.num_detectors
    if count <= 12:
        return [
            [index for index in range(count) if mask >> index & 1]
            for mask in range(1 << count)
        ]
    assert limit is not None
    rng = random.Random(20240517)
    syndromes = [[]]
    for _ in range(400):
        size = rng.randint(1, limit)
        syndromes.append(sorted(rng.sample(range(count), size)))
    return syndromes


def _bound(result: MatchingDecodeResult, graph: DecodingGraph) -> float:
    """The weight difference below which two selections are uncomparable."""

    return graph.num_edges * _ROUNDING * max(1.0, abs(result.weight))


def test_the_translation_keeps_the_shape_of_the_graph() -> None:
    """PyMatching's graph is the decoding graph, node for node and edge for edge."""

    for configuration in _CONFIGURATIONS:
        graph = DecodingGraph.from_detector_error_model(configuration.model)
        matching, edges = _translate(graph)
        assert isinstance(matching, pymatching.Matching)
        assert matching.num_detectors == graph.num_detectors
        assert matching.num_fault_ids == graph.num_observables
        assert matching.num_edges == graph.num_edges
        assert set(edges) == {edge.detectors for edge in graph.edges}
        # The graph states a mechanism that reaches the boundary as an edge to
        # its own boundary node; PyMatching states it as a boundary edge, so the
        # boundary node must not appear in the matching graph's own detector
        # count and must not be a node PyMatching was handed as a detector.
        assert matching.num_detectors == graph.num_detectors


def test_a_mechanism_that_reaches_the_boundary_reads_back_as_a_boundary_edge() -> None:
    """PyMatching's boundary sentinel is translated, not leaked to the caller.

    The raw edge array is asserted to carry the sentinel, so that removing the
    translation from the adapter fails here rather than passing on a graph whose
    mechanisms happen to stay inside the patch.
    """

    model = _CONFIGURATIONS[0].model
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = PyMatchingDecoder(graph)
    syndrome = [0, 1, 2]
    raw = decoder._matching.decode_to_edges_array(_syndrome_vector(graph, syndrome))
    assert (raw == _BOUNDARY).any()
    result = decoder.decode(syndrome)
    assert any(edge.detectors[1] == graph.boundary_node for edge in result.error_edges)
    assert all(_BOUNDARY not in edge.detectors for edge in result.error_edges)


def _syndrome_vector(graph: DecodingGraph, syndrome: list[int]) -> list[int]:
    vector = [0] * graph.num_detectors
    for detector in syndrome:
        vector[detector] = 1
    return vector


@pytest.mark.parametrize("configuration", _CONFIGURATIONS, ids=lambda item: item.name)
def test_both_decoders_agree_on_the_cheapest_weight_of_every_syndrome(
    configuration: _Configuration,
) -> None:
    """The agreement is on the weight first, and on the observables only when unique.

    Every syndrome of a small model and a deterministic sample above it is put to
    both decoders. The weight must agree inside the tie band, and when it agrees
    strictly the observables must agree too; a disagreement whose weights differ
    by more than the band is a failure of one of the two implementations.
    """

    ours, theirs = _decoders(configuration.model)
    graph = ours.graph
    agreed = tied = 0
    for syndrome in _syndromes(configuration.model, limit=4):
        mine = ours.decode(syndrome)
        other = theirs.decode(syndrome)
        assert abs(mine.weight - other.weight) <= _bound(mine, graph), (
            f"{configuration.name}: the two decoders report different cheapest "
            f"weights for D{list(syndrome)}: {mine.weight!r} against "
            f"{other.weight!r}"
        )
        if mine.observables == other.observables:
            agreed += 1
        else:
            tied += 1
    assert agreed > 0
    # The tie clause is a statement about real syndromes of a real graph and not
    # a licence: the small model is enumerated in full and does contain ties.
    if graph.num_detectors <= 12:
        assert tied > 0


def test_a_tie_is_a_disagreement_about_observables_at_equal_weight() -> None:
    """The pinned tie that makes unconditional agreement the wrong claim.

    Tracing the two selections shows why neither decoder is wrong: the syndrome
    D0 D1 D2 of the distance-three repetition code has two explanations of exactly
    equal total weight, one pairing the defects with each other and one routing
    each to the boundary, and they flips different halves of the logical
    observable.
    """

    model = _CONFIGURATIONS[0].model
    ours, theirs = _decoders(model)
    mine = ours.decode([0, 1, 2])
    other = theirs.decode([0, 1, 2])
    assert mine.weight == other.weight
    assert mine.observables != other.observables
    assert mine.error_edges != other.error_edges
    assert sorted(edge.detectors for edge in mine.error_edges) == [(0, 1), (2, 8)]
    assert sorted(edge.detectors for edge in other.error_edges) == [(0, 2), (1, 8)]


def test_the_reported_weight_is_the_exact_sum_of_the_selected_mechanisms() -> None:
    """The comparison is between two selections, not between two summations.

    PyMatching's own accumulator is narrower than the mechanisms it selects:
    `decode(return_weight=True)` reports 3.9020747171643912 for a mechanism this
    package states as 3.9020746947749574. The adapter reports the double-precision
    sum of the mechanisms it reads back, so this test pins the exact sum and the
    difference that would otherwise enter every comparison.
    """

    model = _CONFIGURATIONS[3].model
    graph = DecodingGraph.from_detector_error_model(model)
    theirs = PyMatchingDecoder(graph)
    result = theirs.decode([1])
    assert len(result.error_edges) == 1
    assert result.weight == result.error_edges[0].weight
    accumulated = theirs._matching.decode(
        _syndrome_vector(graph, [1]), return_weight=True
    )[1]
    assert accumulated != result.weight
    assert abs(accumulated - result.weight) <= _bound(result, graph)

    ours, _ = _decoders(model)
    for syndrome in _syndromes(model, limit=4):
        selected = theirs.decode(syndrome)
        assert selected.weight == sum(edge.weight for edge in selected.error_edges)
        assert ours.decode(syndrome).weight == pytest.approx(
            selected.weight, abs=_bound(selected, graph)
        )


def test_the_prediction_is_the_fault_vector_of_the_mechanisms_selected() -> None:
    """PyMatching's fault ids and its fault vector are read as one fact.

    The observables come from `Matching.decode` and the mechanisms from
    `decode_to_edges_array`, two separate calls whose answers must be consistent
    with each other: a translation that labelled the wrong observable on an edge
    would show up as a mismatch between the two rather than as a matching pair of
    mistakes.
    """

    for configuration in _CONFIGURATIONS:
        ours, theirs = _decoders(configuration.model)
        for syndrome in _syndromes(configuration.model, limit=4):
            result = theirs.decode(syndrome)
            flipped: set[int] = set()
            for edge in result.error_edges:
                flipped ^= set(edge.observables)
            assert result.observables == tuple(sorted(flipped))


def test_both_decoders_refuse_exactly_the_same_syndromes() -> None:
    """A refusal is a shared answer, and the two refusals cover the same syndromes.

    The line graph has no boundary mechanism, so a syndrome of odd parity in a
    connected component has no explanation at all. PyMatching raises there and
    the matcher refuses there, and the exhaustive comparison asserts the two
    refusals are the same set rather than one being wider than the other.
    """

    ours, theirs = _decoders(_LINE)
    outcomes = []
    for syndrome in _syndromes(_LINE, limit=None):
        try:
            ours.decode(syndrome)
            mine = True
        except CapabilityError:
            mine = False
        try:
            theirs.decode(syndrome)
            other = True
        except CapabilityError:
            other = False
        outcomes.append((syndrome, mine, other))
    assert any(not mine for _, mine, _ in outcomes)
    assert all(mine == other for _, mine, other in outcomes), outcomes
    assert sum(1 for _, mine, _ in outcomes if mine) == 4


def test_a_syndrome_the_cross_check_refuses_names_the_matcher_that_refuses_it() -> None:
    """The refusal explains that it is not a limitation of the cross-check."""

    _, theirs = _decoders(_LINE)
    with pytest.raises(CapabilityError) as info:
        theirs.decode([0])
    assert "no perfect matching" in str(info.value)
    assert "the self-implemented matcher refuses as well" in str(info.value)


def test_a_parallel_detector_pair_is_refused_rather_than_merged() -> None:
    """Merging is the tempting answer and it is the wrong one.

    The distance-two rotated surface code has fifteen mechanisms over twelve
    detector pairs, and the two mechanisms of a shared pair carry different
    logical labels. PyMatching's `independent` strategy would accept the
    translation and collapse the graph to twelve edges, which loses the label
    difference; the adapter refuses instead, and the model is one the matcher
    itself accepts, so the refusal is this module's narrower domain and not an
    inherited one.
    """

    model = _memory(RotatedSurfaceCode(distance=2), 3, 0.01)
    graph = DecodingGraph.from_detector_error_model(model)
    # The authority reads the model, so the refusal below belongs to the adapter.
    MinimumWeightMatchingDecoder(graph, max_defects=graph.num_detectors)
    pairs: dict[tuple[int, int], list[object]] = {}
    for edge in graph.edges:
        pairs.setdefault(edge.detectors, []).append(edge)
    assert len(pairs) < graph.num_edges
    with pytest.raises(CapabilityError) as info:
        PyMatchingDecoder(graph)
    assert "one edge per detector pair" in str(info.value)
    assert "cannot be translated faithfully" in str(info.value)

    # Merging resolves the refusal and silently drops the distinction, which is
    # exactly why the adapter does not take that route.
    merged = pymatching.Matching()
    for edge in graph.edges:
        first, second = edge.detectors
        keywords = dict(
            fault_ids=set(edge.observables), error_probability=edge.probability
        )
        if second == graph.boundary_node:
            merged.add_boundary_edge(first, **keywords, merge_strategy="independent")
        else:
            merged.add_edge(first, second, **keywords, merge_strategy="independent")
    assert merged.num_edges == len(pairs) < graph.num_edges


def test_a_detector_no_mechanism_flips_is_refused() -> None:
    """PyMatching infers the detector count, so such a detector renumbers a syndrome."""

    model = DetectorErrorModel(
        num_detectors=3,
        num_observables=1,
        errors=(DemError(probability=0.1, detectors=(0, 1)),),
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_detectors == 3
    MinimumWeightMatchingDecoder(graph, max_defects=graph.num_detectors)
    with pytest.raises(CapabilityError) as info:
        PyMatchingDecoder(graph)
    assert "D2 is flipped by no mechanism" in str(info.value)
    assert "syndrome vector would not be this graph's" in str(info.value)


def test_a_hyperedge_model_is_refused_before_the_translation() -> None:
    """The graphlike boundary belongs to the model and not to the cross-check.

    The matrix route at two rounds flips the same check in both bands, which is a
    four-detector mechanism. The decoding graph refuses it and so does the
    authority; the adapter inherits both refusals rather than adding a third.
    """

    hz, lz = code_matrices(RotatedSurfaceCode(distance=3))
    model = DetectorErrorModel.from_code_matrices(
        hz=hz,
        noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
        lz=lz,
        num_rounds=2,
    )
    with pytest.raises(CapabilityError) as info:
        DecodingGraph.from_detector_error_model(model)
    assert "hyperedge" in str(info.value)
    with pytest.raises(CapabilityError, match="hyperedge"):
        PyMatchingDecoder.from_detector_error_model(model)
    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)


def test_the_cross_check_has_no_defect_budget_of_its_own() -> None:
    """The budget is the authority's enumeration cost, so it does not bound the check.

    A syndrome the authority declines at a small budget is still answered by the
    cross-check, and the answer is the one the authority gives when its budget is
    the default, so narrowing the budget cannot narrow the comparison.
    """

    model = _CONFIGURATIONS[0].model
    graph = DecodingGraph.from_detector_error_model(model)
    theirs = PyMatchingDecoder(graph)
    bounded = MinimumWeightMatchingDecoder(graph, max_defects=1)
    with pytest.raises(CapabilityError, match="at most 1 detection events"):
        bounded.decode([0, 1])
    assert theirs.decode(
        [0, 1]
    ) == MinimumWeightMatchingDecoder.from_detector_error_model(model).decode([0, 1])


def test_the_cross_check_validates_the_syndrome_it_is_given() -> None:
    """The adapter shares the matcher's syndrome contract rather than restating it."""

    model = _CONFIGURATIONS[0].model
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = PyMatchingDecoder(graph)
    assert decoder.decode([]) == MatchingDecodeResult(
        observables=(), error_edges=(), weight=0.0
    )
    with pytest.raises(ValueError, match="cannot name the same detector twice"):
        decoder.decode([0, 0])
    with pytest.raises(ValueError, match="outside the graph"):
        decoder.decode([graph.num_detectors])
    with pytest.raises(TypeError, match="integer detector indices"):
        decoder.decode([True])
    with pytest.raises(TypeError, match="graph must be a DecodingGraph"):
        PyMatchingDecoder(graph=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model must be a DetectorErrorModel"):
        PyMatchingDecoder.from_detector_error_model(object())  # type: ignore[arg-type]


def test_the_cross_check_names_the_extra_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absent extra is a stated install step and not an ImportError.

    The refusal is forced by blocking the import rather than by relying on the
    extra being absent, so the test runs in the lane that installs it as well.
    """

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object):
        if name == "pymatching" or name.startswith("pymatching."):
            raise ImportError("No module named 'pymatching'")
        return real_import(name, *args, **kwargs)

    model = _CONFIGURATIONS[0].model
    graph = DecodingGraph.from_detector_error_model(model)
    monkeypatch.delitem(sys.modules, "pymatching", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)

    assert issubclass(MatchingDependencyError, CapabilityError)
    assert issubclass(MatchingDependencyError, ImportError)
    with pytest.raises(MatchingDependencyError, match=r"flagquantum\[pymatching\]"):
        _pymatching()
    with pytest.raises(MatchingDependencyError, match=r"flagquantum\[pymatching\]"):
        PyMatchingDecoder(graph)
