"""Unit coverage for the belief-propagation decoder.

The exchange is verified against a reference it does not share code with: an
exhaustive enumeration of the model. Every mechanism set has a probability, a
detector signature and a logical label, so the most likely explanation of a
syndrome is a number this module computes from the model alone. On a model whose
factor graph is a tree the beliefs are exact, so the decoder must reach that
explanation for every syndrome the model can produce -- and it must refuse the
syndromes the model cannot produce rather than answer them.

Three properties the enumeration alone would not pin are checked beside it. The
first is that a check's own syndrome bit is part of what the check says: a decoder
that dropped it would be transmitting the empty syndrome, whose priors the decoder
already answers, so it could not settle on any other one. The second is that the
matcher and the exchange agree where both apply and disagree only on a shape the
graph cannot carry, which is stated rather than left to be discovered. The third
is the scope the matcher does not have: a mechanism flipping three detectors is a
variable touching three checks here, and the answer is still checked against the
enumeration rather than against the exchange's own opinion of itself.
"""

from __future__ import annotations

import math

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec.belief_propagation import (
    BeliefPropagationDecoder,
    BeliefPropagationDecodeResult,
)
from flagquantum.qec.decoding_graph import DecodingGraph, DecodingGraphEdge
from flagquantum.qec.dem import DemError, DetectorErrorModel
from flagquantum.qec.matching import MinimumWeightMatchingDecoder

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def _chain() -> DetectorErrorModel:
    """A model whose factor graph is a tree: five mechanisms, four checks.

    The mechanisms form a path, so every check carries two variables except at the
    ends, and the beliefs settle on the exact posterior rather than on a fixed
    point of a graph with cycles.
    """

    return _model(
        DemError(probability=0.08, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.12, detectors=(1, 2), observables=(1,)),
        DemError(probability=0.06, detectors=(2, 3), observables=(0,)),
        DemError(probability=0.03, detectors=(0,), observables=(1,)),
        DemError(probability=0.09, detectors=(3,), observables=(0,)),
        detectors=4,
        observables=2,
    )


def _hyperedge() -> DetectorErrorModel:
    """A model the matcher cannot represent: one mechanism flips three detectors."""

    return _model(
        DemError(probability=0.1, detectors=(0, 1, 2), observables=(0,)),
        DemError(probability=0.05, detectors=(0, 1), observables=(1,)),
        DemError(probability=0.05, detectors=(1, 2), observables=(0,)),
        DemError(probability=0.05, detectors=(0, 2), observables=(0,)),
        detectors=3,
        observables=2,
    )


def _most_likely(model: DetectorErrorModel, syndrome: int) -> int | None:
    """The most likely set of mechanisms whose flips are exactly the syndrome.

    ``None`` when no set of mechanisms explains it, which is a property of the
    model: the mechanisms span a subspace of the detector space, and a syndrome
    outside that subspace did not come from this model.
    """

    best: int | None = None
    best_probability = -math.inf
    for mask in range(1 << len(model.errors)):
        flips = 0
        log_probability = 0.0
        for index, error in enumerate(model.errors):
            if mask >> index & 1:
                for detector in error.detectors:
                    flips ^= 1 << detector
                log_probability += math.log(error.probability)
            else:
                log_probability += math.log(1.0 - error.probability)
        if flips == syndrome and log_probability > best_probability:
            best_probability = log_probability
            best = mask
    return best


def _selected(model: DetectorErrorModel, mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(len(model.errors)) if mask >> index & 1)


def _syndromes(model: DetectorErrorModel) -> list[tuple[int, ...]]:
    return [
        tuple(index for index in range(model.num_detectors) if syndrome >> index & 1)
        for syndrome in range(1 << model.num_detectors)
    ]


def _mask(syndrome: tuple[int, ...]) -> int:
    return sum(1 << index for index in syndrome)


def _flips(model: DetectorErrorModel, mechanisms: tuple[int, ...]) -> int:
    flipped = 0
    for mechanism in mechanisms:
        for detector in model.errors[mechanism].detectors:
            flipped ^= 1 << detector
    return flipped


def _weight(model: DetectorErrorModel, mechanisms: tuple[int, ...]) -> float:
    return sum(
        math.log(
            (1.0 - model.errors[mechanism].probability)
            / model.errors[mechanism].probability
        )
        for mechanism in mechanisms
    )


def _observables(
    model: DetectorErrorModel, mechanisms: tuple[int, ...]
) -> tuple[int, ...]:
    """The logical labels a mechanism set flips, exclusive-or like the model's own."""

    flipped: set[int] = set()
    for mechanism in mechanisms:
        for index in model.errors[mechanism].observables:
            if index in flipped:
                flipped.remove(index)
            else:
                flipped.add(index)
    return tuple(sorted(flipped))


def _cheapest(model: DetectorErrorModel, syndrome: int) -> float | None:
    """The smallest total weight of any mechanism set that explains the syndrome.

    A mechanism set that explains the syndrome and is heavier than this is a
    correction and not the most likely one, which is what an approximate exchange
    is allowed to return off a tree -- so this is the floor the decoder's weight is
    checked against rather than an equality.
    """

    floor: float | None = None
    for mask in range(1 << len(model.errors)):
        flips = 0
        weight = 0.0
        for index, error in enumerate(model.errors):
            if mask >> index & 1:
                for detector in error.detectors:
                    flips ^= 1 << detector
                weight += math.log((1.0 - error.probability) / error.probability)
        if flips == syndrome and (floor is None or weight < floor):
            floor = weight
    return floor


def _edge(model: DetectorErrorModel, mechanism: int) -> DecodingGraphEdge:
    detectors = model.errors[mechanism].detectors
    second = detectors[1] if len(detectors) == 2 else model.num_detectors
    return DecodingGraphEdge(
        detectors=(detectors[0], second),
        probability=model.errors[mechanism].probability,
        observables=model.errors[mechanism].observables,
    )


def test_the_empty_syndrome_is_answered_before_the_first_message() -> None:
    """Nothing fired, and the priors already say so.

    A decoder that ran an exchange to learn the empty set would be exchanging
    beliefs about a question its own priors answer, and the iteration count is
    where that shows.
    """

    decoder = BeliefPropagationDecoder.from_detector_error_model(_chain())
    result = decoder.decode(())
    assert result == BeliefPropagationDecodeResult(
        converged=True, observables=(), mechanisms=(), weight=0.0, iterations=0
    )


def test_a_tree_decodes_to_the_most_likely_explanation() -> None:
    """On a tree the beliefs are exact, so they must reach the enumeration's answer.

    This is the decoder's correctness claim, and it is made against the model's own
    distribution: every mechanism set has a probability, so the most likely set
    whose flips are the syndrome is a number computed here rather than a number the
    decoder reports about itself.
    """

    model = _chain()
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    for syndrome in _syndromes(model):
        expected = _most_likely(model, _mask(syndrome))
        assert expected is not None
        result = decoder.decode(syndrome)
        assert result.mechanisms == _selected(model, expected)
        assert result.converged is True
        assert result.weight == pytest.approx(_weight(model, result.mechanisms))
        assert result.observables == _observables(model, result.mechanisms)


def test_a_check_reports_its_own_syndrome_bit() -> None:
    """A check whose detector fired says so, and the answer changes because of it.

    The fallback is turned off here on purpose. The priors alone explain the empty
    syndrome, so a decoder that dropped the check's syndrome bit would be
    transmitting the empty syndrome on every other one, would never settle, and
    would raise instead of answering: on a tree, with the bit in place, every
    syndrome the model can produce must settle on the enumeration's answer.
    """

    model = _chain()
    decoder = BeliefPropagationDecoder.from_detector_error_model(model, osd=False)
    settled = 0
    for syndrome in _syndromes(model):
        expected = _most_likely(model, _mask(syndrome))
        if expected is None:
            continue
        result = decoder.decode(syndrome)
        assert result.converged is True
        assert result.mechanisms == _selected(model, expected)
        settled += 1
    assert settled > 1


def test_the_selected_mechanisms_explain_the_syndrome() -> None:
    """The parity of the selected mechanisms' detectors is the syndrome.

    This is the definition of a correction, and it is checked on a model whose
    factor graph has cycles, where the exchange is approximate and its hard
    decision is not guaranteed to satisfy it -- ordered statistics is what makes it
    hold there, and the ``converged`` flag is what says which of the two answered.
    """

    model = _hyperedge()
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    for syndrome in _syndromes(model):
        result = decoder.decode(syndrome)
        assert _flips(model, result.mechanisms) == _mask(syndrome)
        assert result.weight == pytest.approx(_weight(model, result.mechanisms))


def test_a_mechanism_of_three_detectors_is_read_rather_than_refused() -> None:
    """The matcher refuses a hyperedge; this decoder is for exactly that model.

    The mechanism is a variable touching three checks, so what is checked here is
    that every syndrome is answered with something that flips it and with a weight
    the selection actually has. The exchange is approximate on a graph with cycles,
    so the answer is allowed to be heavier than the cheapest explanation and is not
    allowed to be lighter than it: a lighter one would be a weight that is not the
    sum of the mechanisms it reports.
    """

    model = _hyperedge()
    with pytest.raises(CapabilityError) as excinfo:
        MinimumWeightMatchingDecoder.from_detector_error_model(model)
    assert "hyperedge" in str(excinfo.value)

    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    for syndrome in _syndromes(model):
        result = decoder.decode(syndrome)
        assert _flips(model, result.mechanisms) == _mask(syndrome)
        assert result.observables == _observables(model, result.mechanisms)
        assert result.weight == pytest.approx(_weight(model, result.mechanisms))
        floor = _cheapest(model, _mask(syndrome))
        assert floor is not None
        assert result.weight >= floor - 1e-9


def test_a_cycle_can_settle_on_a_heavier_explanation_than_the_cheapest_one() -> None:
    """Off a tree the exchange is approximate, and this is where that shows.

    Belief propagation is exact on a tree and not beyond one, so on the triangle
    the exchange settles on an explanation that flips the syndrome and is not the
    cheapest of those. This is stated as a property rather than left to be
    discovered, because a caller who needs the cheapest explanation on a loopy
    graph needs the matcher, and a caller who needs a hyperedge read needs this
    decoder and a heavier correction than the minimum.
    """

    model = _hyperedge()
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    heavier = 0
    for syndrome in _syndromes(model):
        result = decoder.decode(syndrome)
        floor = _cheapest(model, _mask(syndrome))
        assert floor is not None
        if result.weight > floor + 1e-9:
            heavier += 1
    assert heavier > 0


def test_the_information_set_is_solved_exactly_when_the_exchange_does_not_settle() -> (
    None
):
    """Ordered statistics answers the syndromes belief propagation cannot.

    A budget of one iteration cannot settle on a syndrome whose priors do not
    already explain it, so the run goes through the fallback, and the fallback
    solves the reduced system exactly rather than returning the last iterate. The
    selection therefore explains the syndrome even though the exchange had not, and
    ``converged`` is false for exactly that reason.
    """

    model = _hyperedge()
    syndrome = (0, 1, 2)
    short = BeliefPropagationDecoder.from_detector_error_model(model, max_iterations=1)
    long = BeliefPropagationDecoder.from_detector_error_model(model)
    one = short.decode(syndrome)
    many = long.decode(syndrome)
    assert one.converged is False
    assert one.iterations == 1
    assert many.converged is True
    for result in (one, many):
        assert _flips(model, result.mechanisms) == 0b111
        assert result.weight == pytest.approx(_weight(model, result.mechanisms))


def test_a_syndrome_the_model_cannot_produce_is_refused() -> None:
    """A detector the mechanisms cannot flip in that parity is not answerable.

    The model's mechanisms span a subspace of the detector space, so this decoder
    refuses a syndrome outside it by name instead of returning the least-bad set of
    mechanisms, which would be a correction that does not explain what it was asked
    about.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.1, detectors=(1, 2), observables=(0,)),
        detectors=3,
    )
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    refused = 0
    for syndrome in _syndromes(model):
        if _most_likely(model, _mask(syndrome)) is None:
            with pytest.raises(CapabilityError) as excinfo:
                decoder.decode(syndrome)
            assert "no set of mechanisms explains" in str(excinfo.value)
            refused += 1
        else:
            assert _flips(model, decoder.decode(syndrome).mechanisms) == _mask(syndrome)
    assert refused > 0


def test_a_model_of_alternatives_is_refused() -> None:
    """A group that fires at most once is a correlation the exchange cannot carry.

    The beliefs assume every mechanism is independent, so reading a group as
    independent would weigh each member as though the others could fire with it.
    The refusal names the reason rather than dropping the structure, and it is
    raised while the caller still holds the model rather than at the first
    syndrome.
    """

    model = _model(
        DemError(probability=0.05, detectors=(0, 1), error_id=0),
        DemError(probability=0.05, detectors=(0,), error_id=0),
        DemError(probability=0.1, detectors=(1,), error_id=1),
        detectors=2,
    )
    assert model.error_ids is not None
    with pytest.raises(CapabilityError) as excinfo:
        BeliefPropagationDecoder.from_detector_error_model(model)
    assert "alternatives" in str(excinfo.value)


def test_a_mechanism_that_can_never_fire_is_refused() -> None:
    """A zero rate is an infinite prior ratio, not a very small belief.

    A mechanism no shot can select carries no evidence, and the ratio the exchange
    starts from is not a number, so the model is refused while the caller holds it
    rather than at the first syndrome either.
    """

    model = _model(
        DemError(probability=0.0, detectors=(0,), observables=(0,)),
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        detectors=2,
    )
    with pytest.raises(CapabilityError) as excinfo:
        BeliefPropagationDecoder.from_detector_error_model(model)
    assert "rate of zero" in str(excinfo.value)


def test_the_ordered_statistics_fallback_can_be_turned_off() -> None:
    """A caller who needs the exact posterior can refuse the fallback instead.

    This is the one case where the flag matters: the exchange did not settle, so
    with the fallback off the decoder states that rather than answering from a
    solve whose result is a correction and not a posterior.
    """

    model = _hyperedge()
    decoder = BeliefPropagationDecoder.from_detector_error_model(
        model, max_iterations=1, osd=False
    )
    with pytest.raises(CapabilityError) as excinfo:
        decoder.decode((0, 1, 2))
    assert "did not settle" in str(excinfo.value)


def test_a_detector_outside_the_model_is_refused() -> None:
    """A syndrome names detectors of this model and no others."""

    model = _chain()
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    with pytest.raises(ValueError) as excinfo:
        decoder.decode((0, model.num_detectors))
    assert "outside the model" in str(excinfo.value)
    with pytest.raises(ValueError):
        decoder.decode((-1,))
    with pytest.raises(TypeError):
        decoder.decode(("D0",))


def test_a_detector_named_twice_states_the_same_set() -> None:
    """A syndrome is a set of detectors, so a repeat is not a multiplicity."""

    model = _chain()
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    assert decoder.decode((0, 1)) == decoder.decode((0, 1, 0))


def test_the_result_record_validates_its_own_fields() -> None:
    """The record is the decoder's answer, so it refuses to be built wrongly."""

    def record(**fields: object) -> BeliefPropagationDecodeResult:
        base: dict[str, object] = {
            "converged": True,
            "observables": (),
            "mechanisms": (),
            "weight": 0.0,
            "iterations": 0,
        }
        base.update(fields)
        return BeliefPropagationDecodeResult(**base)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        record(converged=1)
    with pytest.raises(TypeError):
        record(weight="0")
    with pytest.raises(ValueError):
        record(weight=math.inf)
    with pytest.raises(ValueError):
        record(iterations=-1)
    with pytest.raises(TypeError):
        record(iterations=0.5)
    with pytest.raises(TypeError):
        record(mechanisms=("first",))
    with pytest.raises(ValueError):
        record(observables=(-1,))


def test_the_decoder_refuses_settings_it_cannot_honour() -> None:
    """The iteration limit and the fallback flag are read rather than defaulted."""

    model = _chain()
    with pytest.raises(ValueError):
        BeliefPropagationDecoder.from_detector_error_model(model, max_iterations=0)
    with pytest.raises(TypeError):
        BeliefPropagationDecoder.from_detector_error_model(model, osd=1)
    with pytest.raises(TypeError):
        BeliefPropagationDecoder(model="not a model")  # type: ignore[arg-type]


def test_a_graph_lifts_to_the_model_of_its_own_edges() -> None:
    """The graph carrier reads the graph's edges as mechanisms, boundary included.

    A boundary edge is a mechanism flipping one detector, so the lift is exact for
    what the graph carries, and the edge's observable label survives it. A graph is
    the narrow shape -- graphlike, positive rate -- so the lift is of the graph's
    own edges rather than of whatever model the graph was derived from.
    """

    graph = DecodingGraph(
        num_detectors=3,
        num_observables=1,
        edges=(
            DecodingGraphEdge(detectors=(0, 1), probability=0.05, observables=(0,)),
            DecodingGraphEdge(detectors=(1, 2), probability=0.05),
            DecodingGraphEdge(detectors=(2, 3), probability=0.1, observables=(0,)),
        ),
    )
    decoder = BeliefPropagationDecoder.from_decoding_graph(graph)
    result = decoder.decode((2,))
    assert result.mechanisms == (2,)
    assert result.observables == (0,)
    with pytest.raises(TypeError):
        BeliefPropagationDecoder.from_decoding_graph(_chain())  # type: ignore[arg-type]


def test_a_graphlike_model_gives_the_matcher_and_the_beliefs_one_answer() -> None:
    """Where both families apply they agree, and the agreement is the model's.

    The matcher minimizes total weight and the exchange maximizes likelihood, which
    is the same problem on a model where no two mechanisms share a signature, so a
    tree pins the two to each other rather than to either one's report about itself.
    """

    model = _chain()
    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    beliefs = BeliefPropagationDecoder.from_detector_error_model(model)
    for syndrome in _syndromes(model):
        matched = matcher.decode(syndrome)
        decoded = beliefs.decode(syndrome)
        assert set(matched.error_edges) == {
            _edge(model, mechanism) for mechanism in decoded.mechanisms
        }
        assert matched.observables == decoded.observables
        assert matched.weight == pytest.approx(decoded.weight)


def test_a_mechanism_that_flips_no_detector_is_not_a_graph_edge() -> None:
    """The two families differ on a shape the graph cannot carry.

    A mechanism that flips only an observable changes no syndrome, so it is not a
    graph edge and the matcher cannot represent it. The exchange reads it as a
    variable no check touches, whose belief is its prior, so it stays quiet and
    contributes nothing to the answer -- including to the weight, which is the
    observable-only mechanism's own ratio and would otherwise be charged for a
    detector flip that never happened. This is the one place the families are
    expected to disagree, and it is stated rather than left to be discovered.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, observables=(1,)),
        detectors=1,
        observables=2,
    )
    graph = DecodingGraph.from_detector_error_model(model)
    assert graph.num_edges == 1
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    result = decoder.decode(())
    assert result.mechanisms == ()
    assert result.observables == ()
    assert result.weight == 0.0
    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    assert matcher.decode(()).weight == 0.0


def test_the_import_time_refusal_is_the_family_contract() -> None:
    """A class that cannot be built from every carrier is refused at registration.

    The registry documents three carriers, so a name that could be built from a
    model and not from a graph would be a name whose accepted sources depend on
    which name was asked for. The refusal names the member that is missing, and the
    family the registry already holds is unchanged by it.
    """

    from flagquantum.qec.registry import decoder_names, register_decoder

    class _Partial:
        def decode(self, detection_events: object) -> None: ...

        @classmethod
        def from_detector_error_model(cls, model: object) -> _Partial:
            return cls()

    with pytest.raises(TypeError) as excinfo:
        register_decoder("partial_family")(_Partial)  # type: ignore[arg-type]
    assert "from_decoding_graph" in str(excinfo.value)
    assert "partial_family" not in decoder_names()
    assert "belief_propagation" in decoder_names()


def test_the_registry_builds_the_decoder_from_each_carrier() -> None:
    """A name reaches this decoder from a model, from text, and from a graph."""

    from flagquantum.qec.registry import BELIEF_PROPAGATION_NAME, get_decoder

    model = _chain()
    from_model = get_decoder(BELIEF_PROPAGATION_NAME, model)
    from_text = get_decoder(BELIEF_PROPAGATION_NAME, model.to_stim_text())
    from_graph = get_decoder(
        BELIEF_PROPAGATION_NAME, DecodingGraph.from_detector_error_model(model)
    )
    expected = BeliefPropagationDecoder.from_detector_error_model(model).decode((0,))
    for decoder in (from_model, from_text, from_graph):
        assert isinstance(decoder, BeliefPropagationDecoder)
        assert decoder.decode((0,)) == expected


def test_the_model_is_the_only_input() -> None:
    """The decoder reads a detector error model and nothing else."""

    with pytest.raises(TypeError):
        BeliefPropagationDecoder.from_detector_error_model("detector D0\n")  # type: ignore[arg-type]


def test_the_two_rates_of_a_mechanism_are_one_belief() -> None:
    """A mechanism's prior is its own log-likelihood ratio and not a marginal.

    Two mechanisms that flip the same detector are two variables, so the belief each
    starts from is its own rate; a decoder that folded them into one marginal rate
    before propagating would answer a different model. The check is that the
    reported weight is the sum of the selected mechanisms' own ratios, and that the
    selection is the enumeration's.
    """

    model = _model(
        DemError(probability=0.3, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0,), observables=(1,)),
        detectors=1,
        observables=2,
    )
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    expected = _most_likely(model, 1)
    assert expected is not None
    result = decoder.decode((0,))
    assert result.mechanisms == _selected(model, expected)
    assert result.weight == pytest.approx(_weight(model, result.mechanisms))


def test_every_sampled_syndrome_of_a_repetition_model_agrees_with_the_matcher() -> None:
    """On the model the two families share, they predict the same observables.

    The repetition code's model is graphlike, so it is also the model the matcher
    answers. The syndromes are sampled from the model's own distribution rather
    than enumerated, because the model is large enough that enumeration would be
    the expensive part. The two agree on the logical prediction, which is what a
    caller of either family actually uses, and the exchange never reports a cheaper
    correction than the matcher: its selection explains the syndrome like any other,
    and the matcher's is the cheapest of those, so a smaller number here would mean
    one of the two weights is not the quantity it claims to be.
    """

    from flagquantum.qec.circuit import build_memory_circuit
    from flagquantum.qec.codes import RepetitionCode
    from flagquantum.qec.noise import PhenomenologicalNoise

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RepetitionCode(distance=3), rounds=2),
        noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    )
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    sample = model.dem_sampling(shots=60, seed=7)
    for shot in range(sample.shots):
        syndrome = tuple(
            int(index) for index in sample.detectors[shot].nonzero().flatten()
        )
        decoded = decoder.decode(syndrome)
        matched = matcher.decode(syndrome)
        assert set(decoded.observables) == set(matched.observables)
        assert decoded.weight >= matched.weight - 1e-9
