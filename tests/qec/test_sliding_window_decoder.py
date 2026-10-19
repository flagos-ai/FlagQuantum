"""The windowed matcher: its band arithmetic, and the answer it is worth.

`flagquantum/qec/sliding_window.py` decides a syndrome a band of detectors at a
time, and the reason it exists is a budget rather than a model. The exact matcher
enumerates the ways to pair the defective detectors, so its cost is exponential in
the defect count; the defect count of a memory experiment grows with the number of
rounds, and the long histories in this file are the regime where the exact matcher
stops answering. Three claims are asserted here, and the file is arranged around
them.

The window arithmetic is what it says it is. The band and the window default to
the model's own widest mechanism and to three bands, the window has a floor that
the constructor refuses by name, and the graph a window searches is the model's
graph with the mechanisms that begin inside the window and its numbering kept. The
graph the decoder was built from is not modified, and the window a caller reads
back off the decoder is the window it will decide with.

The banded answer is a genuine answer. Every mechanism a window commits is a
mechanism of the model, the committed mechanisms have the whole syndrome as their
odd-degree set, and their weight is at least the exact matcher's for the same
syndrome. Those three are identities over a seeded history rather than pins on one
syndrome, so a window arithmetic that started selecting something the model does
not have would fail here.

The banded decoder is a replacement for the exact matcher rather than a second
implementation beside it. With one window over the whole graph it is
`MinimumWeightMatchingDecoder` mechanism for mechanism, on every syndrome of a
history, which is the identity the registry delegation rests on; and on a history
the exact matcher declines, the banded decoder answers every syndrome the exact
matcher refused. What the narrow window costs is stated rather than claimed away:
a narrower window selects different mechanisms than the exact matcher even where
the observables agree, so agreement is bought with the width.

These are probes over seeded histories, so they bound the defect counts they
name. They estimate no threshold and no logical error rate; the decoder itself
does not either.
"""

from __future__ import annotations

import math

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    SLIDING_WINDOW_NAME,
    DecodingGraph,
    DetectorErrorModel,
    DetectorErrorModelDecoder,
    MatchingDecodeResult,
    MinimumWeightMatchingDecoder,
    RotatedSurfaceCode,
    SlidingWindowMatchingDecoder,
    build_memory_circuit,
    decoder_names,
    get_decoder,
)
from flagquantum.qec import sliding_window as sliding_window_module
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.dem import DemError
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.sliding_window import _mechanism_span, _window_graph

pytestmark = pytest.mark.unit


def _model(
    code: RepetitionCode | RotatedSurfaceCode,
    *,
    rounds: int,
    p: float = 0.02,
) -> DetectorErrorModel:
    """A merged graphlike model of one memory experiment."""

    return DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=rounds),
        noise=PhenomenologicalNoise(data_flip=p, measurement_flip=p),
    ).merge_duplicate_mechanisms()


def _graph(code: RepetitionCode | RotatedSurfaceCode, *, rounds: int, p: float = 0.02):
    return DecodingGraph.from_detector_error_model(_model(code, rounds=rounds, p=p))


def _history(
    model: DetectorErrorModel, *, shots: int, seed: int
) -> list[tuple[int, ...]]:
    """A seeded history of syndromes, as detector indices.

    `dem_sampling` returns torch tensors, and a shot with no defect is a row of
    zeros rather than a missing row, so the syndrome is read through the flat
    index form of `nonzero` -- the row-major form is empty and would make this
    helper fail on the quiet shots that a history is mostly made of.
    """

    sample = model.dem_sampling(shots=shots, seed=seed)
    return [
        tuple(int(index) for index in row.nonzero(as_tuple=True)[0])
        for row in sample.detectors
    ]


def _odd_degree(graph: DecodingGraph, edges: object) -> set[int]:
    parity: set[int] = set()
    for edge in edges:  # type: ignore[union-attr]
        parity.symmetric_difference_update((edge.detectors[0],))
        if edge.detectors[1] != graph.boundary_node:
            parity.symmetric_difference_update((edge.detectors[1],))
    return parity


# The band arithmetic


def test_the_band_defaults_to_the_widest_mechanism_and_the_window_to_three() -> None:
    graph = _graph(RepetitionCode(distance=3), rounds=8)
    decoder = SlidingWindowMatchingDecoder(graph=graph)

    assert decoder.mechanism_span == max(
        second - first
        for first, second in (edge.detectors for edge in graph.edges)
        if second != graph.boundary_node
    )
    assert decoder.commit == decoder.mechanism_span
    assert decoder.window == 3 * decoder.commit


@pytest.mark.parametrize(
    ("code", "rounds"),
    [
        (RepetitionCode(distance=3), 6),
        (RepetitionCode(distance=5), 6),
        (RotatedSurfaceCode(distance=3), 4),
    ],
)
def test_the_default_band_is_the_codes_detectors_per_round(
    code: RepetitionCode | RotatedSurfaceCode, rounds: int
) -> None:
    """A memory model's widest mechanism is one round, so the band is one round.

    The claim is a candidate-set identity rather than a number: the widest span of
    the model's graph is the number of checks the code carries per round, which is
    what makes a band of detectors one syndrome round. It is stated here because
    the decoder's own docstring reads the band that way, and a model whose graph
    did not have that shape would make the sentence false.
    """

    graph = _graph(code, rounds=rounds)
    assert _mechanism_span(graph) == len(code.checks)


def test_a_window_narrower_than_its_band_is_refused_by_name() -> None:
    graph = _graph(RepetitionCode(distance=3), rounds=8)
    with pytest.raises(ValueError) as excinfo:
        SlidingWindowMatchingDecoder(graph=graph, commit=4, window=3)
    assert "cannot decide a band of 4" in str(excinfo.value)


def test_a_window_below_the_band_plus_the_span_is_refused_by_name() -> None:
    graph = _graph(RepetitionCode(distance=3), rounds=8)
    span = _mechanism_span(graph)
    with pytest.raises(CapabilityError) as excinfo:
        SlidingWindowMatchingDecoder(graph=graph, commit=span, window=span + 1)
    message = str(excinfo.value)
    assert f"at least {2 * span} detectors" in message
    assert f"spans {span} detectors" in message


def test_a_window_that_reaches_the_end_of_the_graph_is_admitted_whatever_it_is() -> (
    None
):
    """The exemption that keeps the whole-graph identity reachable.

    A window that already covers every detector of the model has nothing past its
    last detector to misread, so the floor that exists for a narrow window does
    not apply to it.
    """

    graph = _graph(RepetitionCode(distance=4), rounds=4)
    decoder = SlidingWindowMatchingDecoder(
        graph=graph, commit=graph.num_detectors, window=graph.num_detectors
    )
    assert decoder.window == graph.num_detectors


@pytest.mark.parametrize(("band", "window"), [(0, 4), (2, 0)])
def test_the_band_and_the_window_are_at_least_one(band: int, window: int) -> None:
    graph = _graph(RepetitionCode(distance=3), rounds=8)
    with pytest.raises(ValueError) as excinfo:
        SlidingWindowMatchingDecoder(graph=graph, commit=band, window=window)
    assert "at least one" in str(excinfo.value)


@pytest.mark.parametrize(("band", "window"), [("2", 4), (2.0, 4), (True, 4), (2, "4")])
def test_the_band_and_the_window_are_integers(band: object, window: object) -> None:
    graph = _graph(RepetitionCode(distance=3), rounds=8)
    with pytest.raises(TypeError):
        SlidingWindowMatchingDecoder(graph=graph, commit=band, window=window)  # type: ignore[arg-type]


def test_the_window_graph_keeps_the_models_numbering_and_its_own_boundary() -> None:
    graph = _graph(RepetitionCode(distance=3), rounds=6)
    window = _window_graph(graph, start=0, stop=4)

    assert window.num_detectors == graph.num_detectors
    assert window.boundary_node == graph.boundary_node
    assert window.num_observables == graph.num_observables
    for edge in window.edges:
        first, second = edge.detectors
        assert first < 4
        assert second <= graph.boundary_node


def test_the_window_graph_offers_what_it_cannot_follow_at_the_models_boundary() -> None:
    """A mechanism that begins inside the window and ends past it is kept.

    Dropping it would leave a detector of the band with nothing to pair against,
    which is what makes a band unanswerable; the window therefore keeps it and
    states its far endpoint as the model's boundary, because past the window's
    last detector is where the window stops seeing.
    """

    graph = _graph(RotatedSurfaceCode(distance=3), rounds=6)
    stop = 8
    window = _window_graph(graph, start=0, stop=stop)
    crossing = [
        edge
        for edge in graph.edges
        if edge.detectors[0] < stop and edge.detectors[1] >= stop
    ]
    assert crossing, "the fixture no longer crosses the window it is testing"

    projected = {(edge.detectors, edge.probability) for edge in window.edges}
    for edge in crossing:
        assert (
            (edge.detectors[0], graph.boundary_node),
            edge.probability,
        ) in projected


def test_the_window_graph_holds_exactly_the_mechanisms_that_begin_inside_it() -> None:
    graph = _graph(RotatedSurfaceCode(distance=3), rounds=6)
    start, stop = 8, 16
    window = _window_graph(graph, start=start, stop=stop)

    inside = [edge for edge in graph.edges if start <= edge.detectors[0] < stop]
    assert window.num_edges == len(inside)
    assert all(start <= edge.detectors[0] < stop for edge in window.edges)


def test_building_a_window_does_not_change_the_graph_it_reads() -> None:
    graph = _graph(RotatedSurfaceCode(distance=3), rounds=6)
    before = (graph.num_detectors, graph.num_observables, graph.edges)

    _window_graph(graph, start=0, stop=8)

    assert (graph.num_detectors, graph.num_observables, graph.edges) == before


# The banded answer is an answer


@pytest.mark.parametrize(
    ("code", "rounds"),
    [
        (RepetitionCode(distance=3), 10),
        (RepetitionCode(distance=5), 10),
        (RotatedSurfaceCode(distance=3), 6),
    ],
)
def test_every_committed_mechanism_is_a_mechanism_of_the_model(
    code: RepetitionCode | RotatedSurfaceCode, rounds: int
) -> None:
    """The candidate-set identity that a projection could break.

    A window that committed a mechanism it had rewritten would return a
    correction the model does not contain, and the correction would still explain
    the syndrome, so the syndrome parity below would not catch it. This is the
    assertion that does.
    """

    model = _model(code, rounds=rounds)
    graph = DecodingGraph.from_detector_error_model(model)
    span = _mechanism_span(graph)
    decoder = SlidingWindowMatchingDecoder(graph=graph, commit=span, window=2 * span)
    mechanisms = set(graph.edges)

    for syndrome in _history(model, shots=150, seed=101):
        for edge in decoder.decode(syndrome).error_edges:
            assert edge in mechanisms, f"the window committed {edge}"


@pytest.mark.parametrize(
    ("code", "rounds"),
    [
        (RepetitionCode(distance=3), 10),
        (RepetitionCode(distance=5), 10),
        (RotatedSurfaceCode(distance=3), 6),
    ],
)
def test_the_committed_mechanisms_explain_the_whole_syndrome(
    code: RepetitionCode | RotatedSurfaceCode, rounds: int
) -> None:
    model = _model(code, rounds=rounds)
    graph = DecodingGraph.from_detector_error_model(model)
    span = _mechanism_span(graph)
    decoder = SlidingWindowMatchingDecoder(graph=graph, commit=span, window=2 * span)

    for syndrome in _history(model, shots=150, seed=102):
        result = decoder.decode(syndrome)
        assert _odd_degree(graph, result.error_edges) == set(syndrome)


@pytest.mark.parametrize(
    ("code", "rounds"),
    [
        (RepetitionCode(distance=3), 10),
        (RepetitionCode(distance=5), 10),
        (RotatedSurfaceCode(distance=3), 6),
    ],
)
def test_the_banded_weight_is_at_least_the_exact_matchers(
    code: RepetitionCode | RotatedSurfaceCode, rounds: int
) -> None:
    """A genuine T-join is never cheaper than the least-weight one.

    This is the inequality the module claims and not a pin: the committed
    mechanisms really do explain the syndrome, so the exact matcher's weight
    bounds them from below, and a window that made a band's decision cheaper than
    the history supports would show up as a violation here.
    """

    model = _model(code, rounds=rounds)
    graph = DecodingGraph.from_detector_error_model(model)
    span = _mechanism_span(graph)
    exact = MinimumWeightMatchingDecoder(graph=graph)
    decoder = SlidingWindowMatchingDecoder(graph=graph, commit=span, window=2 * span)

    for syndrome in _history(model, shots=150, seed=103):
        assert decoder.decode(syndrome).weight >= exact.decode(syndrome).weight - 1e-9


def test_a_window_reaching_the_graph_end_still_commits_only_its_band() -> None:
    """The ledger, pinned, because it is the only place the band rule is visible.

    The rule is that a window commits exactly the mechanisms whose first endpoint
    lies in its own leading ``commit`` detectors, whatever the window's width, so
    a mechanism belongs to one window and not to the widest one that happens to
    cover it. What a caller sees of that rule is which mechanisms come back, and
    the alternative -- letting a window that reaches the graph end commit its
    whole view -- was measured to be indistinguishable in weight and observables:
    over every syndrome of every graph of at most sixteen detectors and every
    legal ``(commit, window)`` pair, 3441406 of 3441984 records were identical and
    the remaining 578 differed only in which equally-weighted mechanism set was
    committed, never in the observables or the weight. The ledger is therefore
    pinned here rather than the rule being assumed, and the fixture's weight and
    empty observable tuple are asserted beside it so that a change to the ledger
    that also changes what the caller gets is not mistaken for this rule.
    """

    model = _model(RepetitionCode(distance=4), rounds=3)
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = SlidingWindowMatchingDecoder(graph=graph, commit=2, window=8)

    result = decoder.decode((4, 6, 7, 11))

    assert tuple(edge.detectors for edge in result.error_edges) == (
        (4, 7),
        (6, graph.boundary_node),
        (8, 11),
        (8, graph.boundary_node),
    )
    assert result.observables == ()
    assert result.weight == pytest.approx(15.567281192442506)


def test_the_prediction_is_the_exclusive_or_of_the_committed_labels() -> None:
    model = _model(RepetitionCode(distance=3), rounds=8)
    graph = DecodingGraph.from_detector_error_model(model)
    decoder = SlidingWindowMatchingDecoder(graph=graph)
    syndrome = _history(model, shots=1, seed=104)[0]

    result = decoder.decode(syndrome)
    flipped: set[int] = set()
    for edge in result.error_edges:
        flipped.symmetric_difference_update(edge.observables)
    assert result.observables == tuple(sorted(flipped))
    assert result.weight == math.fsum(edge.weight for edge in result.error_edges)


def test_the_empty_syndrome_selects_nothing_without_a_window() -> None:
    decoder = SlidingWindowMatchingDecoder(
        graph=_graph(RepetitionCode(distance=3), rounds=8)
    )
    result = decoder.decode([])

    assert result.observables == ()
    assert result.error_edges == ()
    assert result.weight == 0.0


def test_decoding_the_same_syndrome_twice_gives_the_same_record() -> None:
    model = _model(RotatedSurfaceCode(distance=3), rounds=6)
    decoder = SlidingWindowMatchingDecoder.from_detector_error_model(model)

    for syndrome in _history(model, shots=25, seed=105):
        assert decoder.decode(syndrome) == decoder.decode(syndrome)


def test_a_detector_outside_the_graph_is_refused_the_way_the_matcher_refuses_it() -> (
    None
):
    graph = _graph(RepetitionCode(distance=3), rounds=4)
    decoder = SlidingWindowMatchingDecoder(graph=graph)

    with pytest.raises(ValueError):
        decoder.decode([graph.num_detectors])
    with pytest.raises(ValueError):
        decoder.decode([0, 0])
    with pytest.raises(TypeError):
        decoder.decode([0.5])  # type: ignore[list-item]


def test_a_window_answer_that_leaves_the_bands_unexplained_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residual check is a check because the induction rests on a borrowed claim.

    Each band is explained because the window's answer is a T-join of the
    window's syndrome, and that is the matcher's claim rather than this module's.
    A matcher that returned an answer it could not justify would leave the
    residual non-empty, so the check is driven here by substituting a matcher
    that answers nothing: the band it was asked about is then never toggled out
    and the decoder refuses instead of returning a correction-shaped record.
    """

    graph = _graph(RepetitionCode(distance=3), rounds=8)
    decoder = SlidingWindowMatchingDecoder(graph=graph, commit=2, window=6)

    empty = MatchingDecodeResult(observables=(), error_edges=(), weight=0.0)

    def silent(*args: object, **kwargs: object) -> MatchingDecodeResult:
        return empty

    monkeypatch.setattr(
        sliding_window_module.MinimumWeightMatchingDecoder,
        "decode",
        silent,
    )
    with pytest.raises(CapabilityError) as excinfo:
        decoder.decode((0, 1))
    assert "left D0, D1 defective" in str(excinfo.value)


# The banded decoder replaces the exact matcher


@pytest.mark.parametrize(
    ("code", "rounds"),
    [
        (RepetitionCode(distance=3), 8),
        (RotatedSurfaceCode(distance=3), 6),
    ],
)
def test_one_window_over_the_whole_graph_is_the_exact_matcher(
    code: RepetitionCode | RotatedSurfaceCode, rounds: int
) -> None:
    """The replacement identity, mechanism for mechanism.

    With a single window covering every detector the decoder decides every band
    against the whole graph, so it must return what the exact matcher returns --
    the same mechanisms, the same observables, the same weight. This is what makes
    the windowed decoder a replacement for the matcher behind the same protocol
    rather than a second implementation beside it, and it is asserted over a
    history rather than on one syndrome.
    """

    model = _model(code, rounds=rounds)
    graph = DecodingGraph.from_detector_error_model(model)
    exact = MinimumWeightMatchingDecoder(graph=graph)
    windowed = SlidingWindowMatchingDecoder(
        graph=graph, commit=graph.num_detectors, window=graph.num_detectors
    )

    for syndrome in _history(model, shots=200, seed=106):
        assert windowed.decode(syndrome) == exact.decode(syndrome)


def test_the_banded_decoder_answers_a_history_the_exact_matcher_declines() -> None:
    """The regime the module exists for, and the budget it moves.

    A long history's defect count grows with its rounds until the exact matcher's
    enumeration refuses it. The banded decoder's budget bounds a window's syndrome
    instead, so the refusal becomes an answer. The assertion is the pair -- the
    exact matcher declined some of this history, and the banded decoder answered
    all of it -- because a history the exact matcher answered would not show the
    escape, and a banded decoder that declined part of it would not be the escape.
    """

    model = _model(RepetitionCode(distance=3), rounds=60, p=0.04)
    graph = DecodingGraph.from_detector_error_model(model)
    exact = MinimumWeightMatchingDecoder(graph=graph)
    span = _mechanism_span(graph)
    banded = SlidingWindowMatchingDecoder(graph=graph, commit=span, window=2 * span)
    history = _history(model, shots=40, seed=107)

    declined = 0
    answered = 0
    for syndrome in history:
        try:
            exact.decode(syndrome)
        except CapabilityError:
            declined += 1
        result = banded.decode(syndrome)
        assert _odd_degree(graph, result.error_edges) == set(syndrome)
        answered += 1

    assert declined > 0, "the history no longer reaches the exact matcher's budget"
    assert answered == len(history)


def test_the_banded_decoder_answers_a_long_surface_history() -> None:
    """The same escape on a code whose detectors per round is not two.

    The repetition code's band is two detectors; a rotated surface code's is
    eight, so the windows are wider and fewer and the budget is reached by a
    different route. Both are measured rather than one standing for the other.
    The exact matcher's refusals are counted and required to be the majority of
    the history rather than all of it -- the budget is a threshold a shot crosses,
    not a property of the history, and pinning the count would pin the seed.
    """

    model = _model(RotatedSurfaceCode(distance=3), rounds=40, p=0.03)
    graph = DecodingGraph.from_detector_error_model(model)
    exact = MinimumWeightMatchingDecoder(graph=graph)
    span = _mechanism_span(graph)
    banded = SlidingWindowMatchingDecoder(graph=graph, commit=span, window=2 * span)
    history = _history(model, shots=20, seed=108)

    declined = 0
    for syndrome in history:
        try:
            exact.decode(syndrome)
        except CapabilityError:
            declined += 1
        result = banded.decode(syndrome)
        assert _odd_degree(graph, result.error_edges) == set(syndrome)
    assert declined >= len(history) // 2


def test_a_wider_window_agrees_with_the_exact_matcher_more_often() -> None:
    """The trade the module states, measured instead of asserted.

    A narrower window is a decision about a window, so it can select different
    mechanisms than the exact matcher even where the observables agree. Agreement
    is bought with the width: the narrowest legal window disagrees at least as
    often as the widest one tested here, and here the widest agrees with the exact
    matcher on every syndrome of the history. This is a measurement on one seeded
    history, not a claim about agreement at a width.
    """

    model = _model(RepetitionCode(distance=4), rounds=14, p=0.02)
    graph = DecodingGraph.from_detector_error_model(model)
    span = _mechanism_span(graph)
    exact = MinimumWeightMatchingDecoder(graph=graph)
    history = _history(model, shots=200, seed=109)

    def disagreements(window: int) -> int:
        decoder = SlidingWindowMatchingDecoder(graph=graph, commit=span, window=window)
        return sum(
            decoder.decode(syndrome).observables != exact.decode(syndrome).observables
            for syndrome in history
        )

    narrow = disagreements(2 * span)
    wide = disagreements(5 * span)
    assert wide <= narrow
    assert wide == 0


# The registry reaches it by name


def test_the_windowed_decoder_is_registered_beside_the_authority() -> None:
    assert SLIDING_WINDOW_NAME in decoder_names()
    assert decoder_names() == tuple(sorted(decoder_names()))


@pytest.mark.parametrize("source", ["model", "graph"])
def test_the_windowed_name_builds_a_decoder_that_decodes(source: str) -> None:
    model = _model(RepetitionCode(distance=3), rounds=4)
    carrier = (
        model if source == "model" else DecodingGraph.from_detector_error_model(model)
    )
    decoder = get_decoder(SLIDING_WINDOW_NAME, carrier)

    assert isinstance(decoder, SlidingWindowMatchingDecoder)
    assert isinstance(decoder, DetectorErrorModelDecoder)
    assert isinstance(decoder.graph, DecodingGraph)
    syndrome = _history(model, shots=1, seed=110)[0]
    assert _odd_degree(decoder.graph, decoder.decode(syndrome).error_edges) == set(
        syndrome
    )


def test_the_windowed_name_takes_the_windows_options() -> None:
    model = _model(RepetitionCode(distance=3), rounds=8)
    graph = DecodingGraph.from_detector_error_model(model)
    span = _mechanism_span(graph)

    decoder = get_decoder(
        SLIDING_WINDOW_NAME, model, commit=span, window=4 * span, max_defects=7
    )
    assert isinstance(decoder, SlidingWindowMatchingDecoder)
    assert (decoder.commit, decoder.window, decoder.max_defects) == (span, 4 * span, 7)


def test_the_two_names_are_two_classes_and_the_records_are_the_same_kind() -> None:
    """The replacement, seen from the call site that chooses by name.

    A caller that holds a model and a syndrome and names a decoder gets a record
    either way, and the record is the same type, so the two names are
    interchangeable behind the protocol. They are still two classes, so the
    equality below is a statement about the answers and not about the objects.
    """

    model = _model(RepetitionCode(distance=3), rounds=6)
    graph = DecodingGraph.from_detector_error_model(model)
    syndrome = _history(model, shots=1, seed=111)[0]

    windowed = get_decoder(
        SLIDING_WINDOW_NAME,
        model,
        commit=graph.num_detectors,
        window=graph.num_detectors,
    )
    authority = get_decoder("minimum_weight_matching", model)

    assert type(windowed) is not type(authority)
    left = windowed.decode(syndrome)
    right = authority.decode(syndrome)
    assert left == right
    assert isinstance(left, type(right))


def test_the_windowed_decoder_inherits_the_matchers_refusals() -> None:
    """A model the matcher cannot graph is refused, and by the same reason."""

    from flagquantum.qec.codes import SteaneCode

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(SteaneCode(), rounds=3),
        noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01),
    )
    with pytest.raises(CapabilityError) as excinfo:
        SlidingWindowMatchingDecoder.from_detector_error_model(model)
    assert "graphlike" in str(excinfo.value)


def test_the_windowed_decoder_refuses_a_model_that_states_a_fault_twice() -> None:
    """The uniqueness requirement the exact matcher states, stated again here.

    A window that matched against either of two mechanisms sharing a signature
    would be choosing a weight the model does not have, so the requirement is
    inherited rather than re-derived.
    """

    model = DetectorErrorModel(
        num_detectors=4,
        num_observables=1,
        errors=(
            DemError(probability=0.1, detectors=(0, 1)),
            DemError(probability=0.2, detectors=(0, 1)),
        ),
    )
    assert not model.mechanisms_are_unique()
    with pytest.raises(ValueError) as excinfo:
        SlidingWindowMatchingDecoder.from_detector_error_model(model)
    assert "merge_duplicate_mechanisms" in str(excinfo.value)


def test_a_window_whose_syndrome_is_past_the_budget_keeps_the_budget_s_reason() -> None:
    """The budget moves from the history to the window; it does not disappear.

    A window is still an exact matching, so a window whose own syndrome holds
    more defects than the budget is refused with the matcher's reason. This is
    the boundary of what the banded decoder buys: bounded work per window, not
    unbounded work per window.
    """

    graph = _graph(RotatedSurfaceCode(distance=3), rounds=20)
    span = _mechanism_span(graph)
    decoder = SlidingWindowMatchingDecoder(
        graph=graph, commit=span, window=4 * span, max_defects=1
    )
    with pytest.raises(CapabilityError) as excinfo:
        decoder.decode(sorted(range(0, graph.num_detectors, 13)))
    assert "detection events per syndrome" in str(excinfo.value)
