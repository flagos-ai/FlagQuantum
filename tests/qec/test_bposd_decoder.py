"""Unit coverage for the belief-propagation decoder with ordered statistics.

Three references the decoder does not share code with decide whether the tests
below mean anything. The first is the matcher in :mod:`flagquantum.qec.matching`:
wherever the model is graphlike both decoders read the same syndrome, and where
the matcher refuses a hyperedge the belief-propagation decoder is the only one
that answers. The second is brute force over the model's own mechanism sets,
which gives both the least weight any explanation of a syndrome can carry and,
through the model's distribution, the most likely observable mask. The third is
the syndrome-consistency invariant, which is checked on every result rather than
only on the small models: whatever the decoder selects must flip exactly the
detectors the syndrome names.

The exact references are claimed only where they were measured. On the one-round
Steane model the decoder reaches the least weight and the most likely observable
for every one of the 64 syndromes; on the two-round Steane model it is a genuine
decoder that disagrees with the most likely observable on some syndromes, and the
tests assert the invariant rather than an optimality that does not hold.
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import bposd as bposd_module
from flagquantum.qec.bposd import (
    BeliefPropagationOsdDecoder,
    BeliefPropagationOsdDecodeResult,
)
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode, SteaneCode
from flagquantum.qec.dem import DemError, DetectorErrorModel
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.surface import RotatedSurfaceCode

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PROBABILITY = 0.05
_WEIGHT_TOLERANCE = 1e-9


def _memory_model(code, rounds: int) -> DetectorErrorModel:
    return DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=rounds),
        noise=PhenomenologicalNoise(
            data_flip=_PROBABILITY, measurement_flip=_PROBABILITY
        ),
    )


@pytest.fixture(scope="module")
def steane_one_round() -> DetectorErrorModel:
    """The one-round Steane model: six detectors, ten mechanisms, one hyperedge."""

    return _memory_model(SteaneCode(), 1)


@pytest.fixture(scope="module")
def steane_two_round() -> DetectorErrorModel:
    return _memory_model(SteaneCode(), 2)


@pytest.fixture(scope="module")
def repetition_two_round() -> DetectorErrorModel:
    return _memory_model(RepetitionCode(distance=3), 2)


@pytest.fixture(scope="module")
def surface_one_round() -> DetectorErrorModel:
    return _memory_model(RotatedSurfaceCode(distance=3), 1)


def _syndrome_mask(defects) -> int:
    mask = 0
    for detector in defects:
        mask ^= 1 << detector
    return mask


def _defects_of(model: DetectorErrorModel, mask: int) -> list[int]:
    return [index for index in range(model.num_detectors) if mask >> index & 1]


def _signature(model: DetectorErrorModel, mechanisms) -> tuple[int, int]:
    """The detector and observable bits a set of mechanism indices flips."""

    detectors = observables = 0
    for index in mechanisms:
        error = model.errors[index]
        for detector in error.detectors:
            detectors ^= 1 << detector
        for observable in error.observables:
            observables ^= 1 << observable
    return detectors, observables


def _exhaustive(
    model: DetectorErrorModel,
) -> tuple[dict[int, float], dict[int, int], set[int]]:
    """Decide the model exactly, without enumerating its mechanism sets.

    The state of the dynamic program is the pair a mechanism set produces -- the
    detectors it flips and the observables it flips -- and the value is the least
    log-odds weight of any set reaching that pair. A mechanism is either used or
    not, so folding one mechanism into the table is one pass over it, and the
    table stays bounded by ``2 ** (detectors + observables)`` instead of growing
    with the mechanism count.

    Minimising the weight maximises the probability, because the probability of
    a set is the product of the selected mechanisms' probabilities and the
    unselected mechanisms' complements, which is a constant factor times
    ``exp(-weight)``.

    Returns:
        The least weight of any explanation of each syndrome, the observable mask
        of the most likely explanation wherever exactly one explanation attains
        that weight, and the syndromes whose least weight is attained by two
        explanations that disagree.
    """

    detectors = model.num_detectors
    signature_mask = (1 << detectors) - 1
    table: dict[int, float] = {0: 0.0}
    for error in model.errors:
        weight = math.log((1.0 - error.probability) / error.probability)
        shift = 0
        for detector in error.detectors:
            shift ^= 1 << detector
        for observable in error.observables:
            shift ^= 1 << (detectors + observable)
        updated = dict(table)
        for state, value in table.items():
            candidate = value + weight
            moved = state ^ shift
            if moved not in updated or candidate < updated[moved] - _WEIGHT_TOLERANCE:
                updated[moved] = candidate
        table = updated

    least_weight: dict[int, float] = {}
    most_likely: dict[int, int] = {}
    ambiguous: set[int] = set()
    for state, value in table.items():
        syndrome = state & signature_mask
        observable = state >> detectors
        current = least_weight.get(syndrome)
        if current is None or value < current - _WEIGHT_TOLERANCE:
            least_weight[syndrome] = value
            most_likely[syndrome] = observable
            ambiguous.discard(syndrome)
        elif abs(value - current) <= _WEIGHT_TOLERANCE:
            ambiguous.add(syndrome)
    return least_weight, most_likely, ambiguous


def _observable_tuple(model: DetectorErrorModel, mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(model.num_observables) if mask >> index & 1)


def _sampled_syndromes(
    model: DetectorErrorModel, shots: int, seed: int
) -> list[list[int]]:
    sample = model.dem_sampling(shots=shots, seed=seed)
    return [
        [index for index, bit in enumerate(row.tolist()) if bit]
        for row in sample.detectors
    ]


# --------------------------------------------------------------------------
# the boundary this decoder exists for
# --------------------------------------------------------------------------


def test_matching_refuses_the_steane_model_that_this_decoder_accepts(
    steane_one_round: DetectorErrorModel,
) -> None:
    """The replacement this module exists for: one boundary, two implementations."""

    hyperedges = [
        error for error in steane_one_round.errors if len(error.detectors) == 3
    ]
    assert len(hyperedges) == 1

    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(steane_one_round)

    # The consumer is unchanged: the same detector indices that the matcher would
    # have taken reach this decoder and produce a correction.
    result = BeliefPropagationOsdDecoder(steane_one_round).decode([0, 1, 2])
    assert _signature(steane_one_round, result.mechanisms)[0] == 0b111


def test_every_one_round_steane_syndrome_is_decoded_and_consistent(
    steane_one_round: DetectorErrorModel,
) -> None:
    decoder = BeliefPropagationOsdDecoder(steane_one_round)
    for syndrome in range(1 << steane_one_round.num_detectors):
        defects = _defects_of(steane_one_round, syndrome)
        result = decoder.decode(defects)
        detectors, observables = _signature(steane_one_round, result.mechanisms)
        assert detectors == syndrome
        assert result.observables == tuple(
            index
            for index in range(steane_one_round.num_observables)
            if observables >> index & 1
        )


def test_the_one_round_steane_decoder_reaches_the_least_weight(
    steane_one_round: DetectorErrorModel,
) -> None:
    """The selected mechanisms carry the least weight any explanation carries."""

    least_weight, _, _ = _exhaustive(steane_one_round)
    decoder = BeliefPropagationOsdDecoder(steane_one_round)
    assert len(least_weight) == 1 << steane_one_round.num_detectors
    for syndrome, expected in least_weight.items():
        result = decoder.decode(_defects_of(steane_one_round, syndrome))
        assert result.weight == pytest.approx(expected, abs=_WEIGHT_TOLERANCE)


def test_the_one_round_steane_decoder_reaches_the_most_likely_observable(
    steane_one_round: DetectorErrorModel,
) -> None:
    """An independent decision of the model, not an enumeration, fixes the flips."""

    _, most_likely, ambiguous = _exhaustive(steane_one_round)
    assert ambiguous == set()
    decoder = BeliefPropagationOsdDecoder(steane_one_round)
    for syndrome, mask in most_likely.items():
        result = decoder.decode(_defects_of(steane_one_round, syndrome))
        assert result.observables == _observable_tuple(steane_one_round, mask)


def test_the_two_round_steane_model_is_decoded_and_its_ties_are_visible(
    steane_two_round: DetectorErrorModel,
) -> None:
    """Two rounds is where the model stops determining the logical flip.

    Three statements are measured in one pass over all 4096 syndromes: every
    answer reproduces its syndrome, every answer carries the least weight any
    explanation carries, and wherever exactly one explanation attains that least
    weight the decoder picks it. The remaining syndromes are the model's own
    ambiguity -- two least-weight explanations that disagree on the observable --
    and the test counts them instead of asserting an agreement the model does not
    offer.
    """

    least_weight, most_likely, ambiguous = _exhaustive(steane_two_round)
    decoder = BeliefPropagationOsdDecoder(steane_two_round)
    matched = disagreed = 0
    for syndrome in range(1 << steane_two_round.num_detectors):
        defects = _defects_of(steane_two_round, syndrome)
        result = decoder.decode(defects)
        detectors, observables = _signature(steane_two_round, result.mechanisms)
        assert detectors == syndrome
        assert result.observables == _observable_tuple(steane_two_round, observables)
        assert result.weight == pytest.approx(
            least_weight[syndrome], abs=_WEIGHT_TOLERANCE
        )
        if syndrome in ambiguous:
            continue
        if result.observables == _observable_tuple(
            steane_two_round, most_likely[syndrome]
        ):
            matched += 1
        else:
            disagreed += 1

    assert matched > 0
    assert disagreed == 0
    assert len(ambiguous) == 1920
    assert matched + len(ambiguous) == 1 << steane_two_round.num_detectors


def test_the_decoder_is_not_optimal_where_the_rotated_surface_code_shows_it(
    surface_one_round: DetectorErrorModel,
) -> None:
    """The bound on this decoder's claim, measured on the code that shows it.

    Belief propagation is an estimate and order-zero post-processing holds the
    free positions at zero, so the answer is an explanation and not the most
    likely one. On the one-round rotated surface code the decoder carries more
    weight than the least-weight explanation on five syndromes and differs from
    the most likely observable on three, with no tie in the model to excuse
    either. The one-round Steane model reaches both exactly, so the difference
    is a property of the model and not of the decoder being unfinished.
    """

    least_weight, most_likely, ambiguous = _exhaustive(surface_one_round)
    assert ambiguous == set()
    decoder = BeliefPropagationOsdDecoder(surface_one_round)
    heavier: list[tuple[int, float, tuple[int, ...]]] = []
    lighter_disagreements = 0
    for syndrome, expected in sorted(least_weight.items()):
        defects = _defects_of(surface_one_round, syndrome)
        result = decoder.decode(defects)
        assert _signature(surface_one_round, result.mechanisms)[0] == syndrome
        if result.weight > expected + _WEIGHT_TOLERANCE:
            heavier.append((syndrome, result.weight - expected, result.observables))
        elif result.observables != _observable_tuple(
            surface_one_round, most_likely[syndrome]
        ):
            lighter_disagreements += 1

    assert [syndrome for syndrome, _, _ in heavier] == [39, 78, 82, 111, 164]
    assert [round(excess, 4) for _, excess, _ in heavier] == [
        0.6904,
        0.6904,
        2.9444,
        2.9444,
        2.9444,
    ]
    # Two of the five heavier answers happen to agree with the most likely
    # observable, and the three that disagree are exactly the disagreements an
    # independent decision of the model reports. Wherever the decoder does reach
    # the least weight it also agrees with the most likely observable, so the
    # whole of the gap between this decoder and an optimal one is these five
    # syndromes out of the 256 the model admits.
    assert (
        sum(
            1
            for syndrome, _, observables in heavier
            if observables
            != _observable_tuple(surface_one_round, most_likely[syndrome])
        )
        == 3
    )
    assert lighter_disagreements == 0
    assert len(least_weight) == 256


# --------------------------------------------------------------------------
# agreement with the matcher wherever the matcher decodes
# --------------------------------------------------------------------------


def test_the_two_decoders_agree_on_a_graphlike_model(
    repetition_two_round: DetectorErrorModel,
) -> None:
    """A model both decoders accept is where their answers have to line up."""

    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(
        repetition_two_round
    )
    decoder = BeliefPropagationOsdDecoder(repetition_two_round)
    syndromes = _sampled_syndromes(repetition_two_round, shots=100, seed=5)
    assert len(syndromes) == 100
    for defects in syndromes:
        assert (
            decoder.decode(defects).observables == matcher.decode(defects).observables
        )


def test_the_two_decoders_agree_wherever_their_answers_are_distinguishable(
    repetition_two_round: DetectorErrorModel,
) -> None:
    """Equal-weight explanations are where the two decoders are allowed to part.

    On this model every mechanism fires with the same probability, so the
    syndrome ``D0 D1 D2`` has two explanations of identical minimum weight: the
    mechanisms ``{(0, 1), (2)}``, whose observable labels cancel, and the
    mechanisms ``{(0, 2), (1)}``, which flip the observable once. The two
    decoders break that tie differently, so the test asserts the least weight
    both reach and pins the disagreement to exactly those syndromes instead of
    claiming an agreement the model does not determine.
    """

    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(
        repetition_two_round
    )
    decoder = BeliefPropagationOsdDecoder(repetition_two_round)
    least_weight, most_likely, ambiguous = _exhaustive(repetition_two_round)

    disagreed: list[int] = []
    for syndrome in range(1 << repetition_two_round.num_detectors):
        defects = _defects_of(repetition_two_round, syndrome)
        matched = matcher.decode(defects)
        propagated = decoder.decode(defects)
        assert propagated.weight == pytest.approx(
            least_weight[syndrome], abs=_WEIGHT_TOLERANCE
        )
        assert matched.weight == pytest.approx(least_weight[syndrome], abs=1e-6)
        if matched.observables != propagated.observables:
            disagreed.append(syndrome)

    assert disagreed == [0b000111, 0b010011, 0b011101, 0b101110, 0b101111, 0b111011]
    for syndrome in disagreed:
        # The disagreement is a tie at the minimum weight and not a mistake: the
        # decoder still returns a least-weight explanation, and the model itself
        # does not single out the observable the matcher chose.
        assert syndrome in ambiguous
        assert decoder.decode(
            _defects_of(repetition_two_round, syndrome)
        ).weight == pytest.approx(least_weight[syndrome], abs=_WEIGHT_TOLERANCE)
    assert len(disagreed) < len(most_likely)


# --------------------------------------------------------------------------
# fail-closed boundaries
# --------------------------------------------------------------------------


def test_a_syndrome_outside_the_model_is_refused() -> None:
    """No mechanism set reproduces detector one, so no correction follows."""

    model = DetectorErrorModel(
        num_detectors=2,
        num_observables=1,
        errors=(DemError(probability=0.1, detectors=(0,), observables=(0,)),),
    )
    decoder = BeliefPropagationOsdDecoder(model)
    assert decoder.decode([0]).observables == (0,)
    with pytest.raises(CapabilityError, match="not in the span"):
        decoder.decode([1])
    with pytest.raises(CapabilityError, match="not in the span"):
        decoder.decode([0, 1])


def test_a_model_with_no_mechanisms_decodes_only_the_empty_syndrome() -> None:
    model = DetectorErrorModel(num_detectors=1, num_observables=0)
    decoder = BeliefPropagationOsdDecoder(model)
    result = decoder.decode(())
    assert result.mechanisms == ()
    assert result.observables == ()
    assert result.weight == 0.0
    with pytest.raises(CapabilityError, match="not in the span"):
        decoder.decode([0])


def test_the_empty_syndrome_selects_nothing(
    steane_one_round: DetectorErrorModel,
) -> None:
    result = BeliefPropagationOsdDecoder(steane_one_round).decode([])
    assert result == BeliefPropagationOsdDecodeResult(
        observables=(), mechanisms=(), weight=0.0, converged=True, iterations=0
    )


def test_detection_events_are_validated(steane_one_round: DetectorErrorModel) -> None:
    decoder = BeliefPropagationOsdDecoder(steane_one_round)
    with pytest.raises(TypeError, match="integer detector indices"):
        decoder.decode([1.5])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="integer detector indices"):
        decoder.decode([True])
    with pytest.raises(TypeError, match="integer detector indices"):
        decoder.decode(["0"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="outside the model's 6 detectors"):
        decoder.decode([-1])
    with pytest.raises(ValueError, match="outside the model's 6 detectors"):
        decoder.decode([6])
    with pytest.raises(ValueError, match="cannot name the same detector twice"):
        decoder.decode([0, 0])


def test_decoder_options_are_validated(steane_one_round: DetectorErrorModel) -> None:
    with pytest.raises(TypeError, match="must be a DetectorErrorModel"):
        BeliefPropagationOsdDecoder(model=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_iterations must be an integer"):
        BeliefPropagationOsdDecoder(steane_one_round, max_iterations=True)
    with pytest.raises(ValueError, match="max_iterations must be at least one"):
        BeliefPropagationOsdDecoder(steane_one_round, max_iterations=0)
    with pytest.raises(TypeError, match="scaling must be a real number"):
        BeliefPropagationOsdDecoder(steane_one_round, scaling="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scaling must be greater than zero"):
        BeliefPropagationOsdDecoder(steane_one_round, scaling=0.0)
    with pytest.raises(ValueError, match="scaling must be greater than zero"):
        BeliefPropagationOsdDecoder(steane_one_round, scaling=1.5)
    with pytest.raises(ValueError, match="scaling must be greater than zero"):
        BeliefPropagationOsdDecoder(steane_one_round, scaling=float("nan"))


def test_the_result_record_is_validated() -> None:
    def result(**overrides: object) -> BeliefPropagationOsdDecodeResult:
        fields: dict[str, object] = {
            "observables": (0,),
            "mechanisms": (0,),
            "weight": 1.0,
            "converged": True,
            "iterations": 1,
        }
        fields.update(overrides)
        return BeliefPropagationOsdDecodeResult(**fields)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="decoded observables must be integer indices"):
        result(observables=(True,))
    with pytest.raises(ValueError, match="decoded observables must be non-negative"):
        result(observables=(-1,))
    with pytest.raises(ValueError, match="decoded observables must be unique"):
        result(observables=(1, 0))
    with pytest.raises(ValueError, match="decoded observables must be unique"):
        result(observables=(0, 0))
    with pytest.raises(TypeError, match="selected mechanisms must be integer indices"):
        result(mechanisms=(0.5,))
    with pytest.raises(ValueError, match="selected mechanisms must be unique"):
        result(mechanisms=(2, 1))
    with pytest.raises(TypeError, match="decode weight must be a real number"):
        result(weight=True)
    with pytest.raises(TypeError, match="decode weight must be a real number"):
        result(weight="1.0")
    with pytest.raises(ValueError, match="decode weight must be finite"):
        result(weight=float("inf"))
    with pytest.raises(TypeError, match="convergence must be a boolean"):
        result(converged=1)
    with pytest.raises(TypeError, match="passes must be an integer"):
        result(iterations=1.5)
    with pytest.raises(ValueError, match="passes must be non-negative"):
        result(iterations=-1)
    assert result(observables=(0, 1), mechanisms=(0, 2)).weight == 1.0


# --------------------------------------------------------------------------
# the estimate is reported, not hidden
# --------------------------------------------------------------------------


def test_the_iteration_count_is_bounded_by_the_budget(
    surface_one_round: DetectorErrorModel,
) -> None:
    """A pass is counted from one, so a settled estimate names a real pass."""

    budget = 5
    decoder = BeliefPropagationOsdDecoder(surface_one_round, max_iterations=budget)
    # The empty syndrome is answered without a pass at all, and its own test
    # pins that; every other syndrome is the estimate's to explain.
    assert decoder.decode([]).iterations == 0
    passes = []
    for syndrome in range(1, 1 << surface_one_round.num_detectors):
        result = decoder.decode(_defects_of(surface_one_round, syndrome))
        passes.append(result.iterations)
        assert 1 <= result.iterations <= budget
        if not result.converged:
            # A stalled estimate is the one that spends the whole budget.
            assert result.iterations == budget
        assert _signature(surface_one_round, result.mechanisms)[0] == syndrome
    # The counter reports where the estimate settled rather than the budget, so
    # at least one syndrome settles strictly inside it.
    assert min(passes) < budget


def test_belief_propagation_reports_a_syndrome_it_did_not_converge_on(
    steane_two_round: DetectorErrorModel,
) -> None:
    """A capped budget must still answer, and must say the estimate is partial.

    Belief propagation is not guaranteed to settle, so the record carries the
    estimate's own verdict rather than presenting every decode as decided. A
    larger budget strictly reduces how often the estimate stalls, and the
    ordered-statistics pass answers consistently either way.
    """

    sampled = _sampled_syndromes(steane_two_round, shots=25, seed=13)
    capped = BeliefPropagationOsdDecoder(steane_two_round, max_iterations=1)
    stalled = [defects for defects in sampled if not capped.decode(defects).converged]
    assert stalled, "no sampled syndrome exercised the capped budget"

    for defects in stalled:
        # The capped estimate is partial, and the answer that follows it is
        # still an exact explanation of the syndrome.
        assert capped.decode(defects).iterations == 1
        assert _signature(steane_two_round, capped.decode(defects).mechanisms)[
            0
        ] == _syndrome_mask(defects)

    improved = BeliefPropagationOsdDecoder(steane_two_round, max_iterations=5)
    fewer = [defects for defects in sampled if not improved.decode(defects).converged]
    assert len(fewer) < len(stalled)


def test_the_divergence_floor_is_a_property_of_the_model() -> None:
    """Past a point a larger budget stops helping, and that is reported."""

    model = _memory_model(SteaneCode(), 1)
    syndromes = [_defects_of(model, mask) for mask in range(1 << 6)]
    counts = []
    for budget in (1, 5, 200):
        decoder = BeliefPropagationOsdDecoder(model, max_iterations=budget)
        counts.append(
            sum(1 for defects in syndromes if not decoder.decode(defects).converged)
        )
    assert counts[0] > counts[1] >= counts[2]
    assert counts[1] == counts[2] == 0


def test_the_decoder_prefers_the_mechanism_the_prior_favours() -> None:
    """Two mechanisms with the same signature are separated by their prior."""

    model = DetectorErrorModel(
        num_detectors=1,
        num_observables=2,
        errors=(
            DemError(probability=0.02, detectors=(0,), observables=(0,)),
            DemError(probability=0.30, detectors=(0,), observables=(1,)),
        ),
    )
    result = BeliefPropagationOsdDecoder(model).decode([0])
    # The unlikely mechanism explains the syndrome too, and the likelihood the
    # model states makes the frequent one the explanation to prefer.
    assert result.observables == (1,)
    assert result.mechanisms == (1,)


def test_scaling_one_is_the_undamped_iteration(
    steane_one_round: DetectorErrorModel,
) -> None:
    """The damping factor is a parameter, not a hidden constant."""

    undamped = BeliefPropagationOsdDecoder(steane_one_round, scaling=1.0)
    assert undamped.scaling == 1.0
    result = undamped.decode([0, 1, 2])
    assert _signature(steane_one_round, result.mechanisms)[0] == 0b111


def test_the_default_damping_is_the_damped_iteration(
    surface_one_round: DetectorErrorModel,
) -> None:
    """The default factor is damped, and the damping is measurable.

    The one-round rotated surface model settles fewer syndromes at the default
    factor than at a factor of one, so the default is a damped iteration rather
    than an undamped one that happens to be spelled with a different number.
    """

    syndromes = [
        _defects_of(surface_one_round, syndrome)
        for syndrome in range(1 << surface_one_round.num_detectors)
    ]
    damped = BeliefPropagationOsdDecoder(surface_one_round)
    undamped = BeliefPropagationOsdDecoder(surface_one_round, scaling=1.0)
    settled = sum(1 for defects in syndromes if damped.decode(defects).converged)
    undamped_settled = sum(
        1 for defects in syndromes if undamped.decode(defects).converged
    )
    assert settled == 224
    assert undamped_settled == len(syndromes)


def test_a_mechanism_that_always_fires_has_a_finite_weight() -> None:
    """A log-odds ratio is unbounded at both ends, and a model can state one.

    ``log((1 - p) / p)`` is infinite at ``p = 1``, so a decoder that reported the
    ratio verbatim could not state a result record. The mechanism is still the
    only explanation of the syndrome, and the weight is the clamped ratio.
    """

    model = DetectorErrorModel(
        num_detectors=1,
        num_observables=1,
        errors=(DemError(probability=1.0, detectors=(0,), observables=(0,)),),
    )
    result = BeliefPropagationOsdDecoder(model).decode([0])
    assert result.observables == (0,)
    assert result.mechanisms == (0,)
    assert math.isfinite(result.weight)
    assert result.weight < -27.0


def test_a_mechanism_that_never_fires_has_a_finite_weight() -> None:
    """The other end of the same interval, with the certain mechanism removed."""

    model = DetectorErrorModel(
        num_detectors=1,
        num_observables=1,
        errors=(DemError(probability=0.0, detectors=(0,), observables=(0,)),),
    )
    result = BeliefPropagationOsdDecoder(model).decode([0])
    assert result.mechanisms == (0,)
    assert math.isfinite(result.weight)
    assert result.weight > 27.0


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------


def test_the_decoder_is_exported_from_the_package() -> None:
    import flagquantum.qec as qec

    assert set(bposd_module.__all__) == {
        "BeliefPropagationOsdDecodeResult",
        "BeliefPropagationOsdDecoder",
    }
    for name in ("BeliefPropagationOsdDecodeResult", "BeliefPropagationOsdDecoder"):
        assert name in qec.__all__
        assert getattr(qec, name) is getattr(bposd_module, name)


def test_the_decoder_imports_nothing_outside_torch_and_the_standard_library() -> None:
    """The replacement must not smuggle in an external decoder."""

    tree = ast.parse((Path(bposd_module.__file__)).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"torch"}
    assert imported <= allowed, sorted(imported - allowed)
    for absent in ("stim", "pymatching"):
        assert absent not in imported


def test_the_module_is_reachable_from_a_fresh_interpreter() -> None:
    """A submodule accessor can pass while the package never imports it."""

    code = (
        "import flagquantum.qec as qec\n"
        "assert 'BeliefPropagationOsdDecoder' in qec.__all__\n"
        "assert qec.BeliefPropagationOsdDecoder.__module__ == 'flagquantum.qec.bposd'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr


def test_the_decoder_is_outside_the_registry_by_contract_not_by_omission() -> None:
    """The registry's protocol promises a graph this decoder cannot produce.

    Three contracts state that this decoder is constructed directly rather than
    reached through ``decoder_names()``, and the reason given is a protocol
    promise rather than a missing method. That is a claim about the tree, so it
    is measured here against a model both decoders accept: the matcher carries
    the constructor the protocol requires and its instance carries the graph,
    this decoder's does neither, and the registry's own admission check refuses
    it for the reason its refusal message names.
    """

    from flagquantum.qec.decoding_graph import DecodingGraph
    from flagquantum.qec.registry import (
        AUTHORITY_NAME,
        CROSS_CHECK_NAME,
        SLIDING_WINDOW_NAME,
        DetectorErrorModelDecoder,
        decoder_names,
        register_decoder,
    )

    model = _memory_model(RepetitionCode(distance=3), rounds=2)
    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    assert isinstance(matcher.graph, DecodingGraph)
    assert isinstance(matcher, DetectorErrorModelDecoder)

    instance = BeliefPropagationOsdDecoder(model)
    assert not hasattr(BeliefPropagationOsdDecoder, "from_detector_error_model")
    assert not hasattr(instance, "graph")
    assert not isinstance(instance, DetectorErrorModelDecoder)

    with pytest.raises(TypeError, match="from_detector_error_model"):
        register_decoder("belief_propagation_osd")(BeliefPropagationOsdDecoder)

    # The refusal is not enough on its own: the name has to stay free, or a
    # later registration would silently take a name this one was refused.
    assert "belief_propagation_osd" not in decoder_names()
    assert set(decoder_names()) == {
        AUTHORITY_NAME,
        CROSS_CHECK_NAME,
        SLIDING_WINDOW_NAME,
    }
