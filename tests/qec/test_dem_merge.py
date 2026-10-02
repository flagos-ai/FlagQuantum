"""The stated merge rule that gives one prior to a shared mechanism signature."""

from __future__ import annotations

import math

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    DemError,
    DemMergeRule,
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    RepetitionCode,
    build_memory_circuit,
)
from flagquantum.qec.noise import PhenomenologicalNoise

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def _shared(*probabilities: float, **shape: int):
    """A model whose mechanisms all flip the one signature ``D0 L0``."""

    return _model(
        *(
            DemError(probability=probability, detectors=(0,), observables=(0,))
            for probability in probabilities
        ),
        **shape,
    )


def _parity(*probabilities: float) -> float:
    """The probability that an odd number of independent mechanisms fire.

    Written as a product of ``1 - 2p`` factors rather than as a fold of the
    module's own arithmetic, so the expected value is derived here instead of
    being read back out of the code under test.
    """

    complement = 1.0
    for probability in probabilities:
        complement *= 1.0 - 2.0 * probability
    return (1.0 - complement) / 2.0


def test_the_rule_has_the_two_stated_members() -> None:
    """The rule is a value a caller states, not a boolean on the operation."""

    assert {member.name for member in DemMergeRule} == {
        "INDEPENDENT_PARITY",
        "CLAMPED_LINEAR_SUM",
    }
    assert {member.value for member in DemMergeRule} == {
        "independent_parity",
        "clamped_linear_sum",
    }
    assert isinstance(DemMergeRule.INDEPENDENT_PARITY, str)


def test_the_parity_rule_is_the_probability_that_an_odd_number_fire() -> None:
    """``p1 + p2 - 2 p1 p2`` for two mechanisms, and one product for three.

    Two mechanisms are one fault to a decoder, which sees their parity. Their
    sum, their mean, and the larger of the two would each give a weight no
    fault has, so the formula is the substance of the rule rather than a
    detail: at ``1/10`` and ``2/10`` the parity is ``0.26``, the sum is
    ``0.3``, and the mean is ``0.15``.
    """

    merged = _shared(0.1, 0.2).merge_duplicate_mechanisms()
    assert merged.errors[0].probability == pytest.approx(0.1 + 0.2 - 2 * 0.1 * 0.2)
    assert merged.errors[0].probability == pytest.approx(0.26)

    wider = _shared(0.1, 0.2, 0.05).merge_duplicate_mechanisms()
    assert wider.num_errors == 1
    assert wider.errors[0].probability == pytest.approx(_parity(0.1, 0.2, 0.05))
    # The same value spelled out as the four disjoint ways an odd number fire.
    assert wider.errors[0].probability == pytest.approx(
        0.1 * 0.8 * 0.95 + 0.2 * 0.9 * 0.95 + 0.05 * 0.9 * 0.8 + 0.1 * 0.2 * 0.05
    )


def test_the_sum_rule_adds_and_saturates_at_one() -> None:
    """``min(1, sum(p))``, so it is a different number and not a spelling."""

    merged = _shared(0.1, 0.2).merge_duplicate_mechanisms(
        rule=DemMergeRule.CLAMPED_LINEAR_SUM
    )
    assert merged.errors[0].probability == pytest.approx(0.3)

    saturated = _shared(0.6, 0.7).merge_duplicate_mechanisms(
        rule=DemMergeRule.CLAMPED_LINEAR_SUM
    )
    assert saturated.errors[0].probability == pytest.approx(1.0)


def test_the_parity_rule_stays_below_one_half_where_the_sum_rule_does_not() -> None:
    """Closure matters: above one half a mechanism has no matching weight."""

    priors = (0.2, 0.2, 0.2, 0.2)
    parity = _shared(*priors).merge_duplicate_mechanisms()
    summed = _shared(*priors).merge_duplicate_mechanisms(
        rule=DemMergeRule.CLAMPED_LINEAR_SUM
    )
    assert parity.errors[0].probability < 0.5
    assert summed.errors[0].probability == pytest.approx(0.8)


def test_a_model_with_distinct_signatures_is_unique() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.1, detectors=(1,), observables=()),
    )
    assert model.mechanisms_are_unique()


def test_a_model_holding_no_mechanisms_is_unique() -> None:
    assert _model().mechanisms_are_unique()


def test_a_shared_signature_is_not_unique() -> None:
    assert not _shared(0.1, 0.2).mechanisms_are_unique()


def test_a_signature_is_both_its_detectors_and_its_observables() -> None:
    """Two mechanisms agreeing on one half of the signature are still distinct."""

    detectors_only = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=()),
        DemError(probability=0.2, detectors=(0, 1), observables=(0,)),
    )
    observables_only = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
    )
    assert detectors_only.mechanisms_are_unique()
    assert observables_only.mechanisms_are_unique()


def test_an_unnormalized_signature_matches_its_normalized_form() -> None:
    """Sharing is decided on the normalized signature, which is what a decoder reads."""

    model = DetectorErrorModel(
        num_detectors=2,
        num_observables=1,
        errors=(
            DemError(probability=0.1, detectors=(1, 0), observables=(0, 0)),
            DemError(probability=0.2, detectors=(0, 1), observables=(0,)),
        ),
    )
    assert not model.mechanisms_are_unique()
    assert model.merge_duplicate_mechanisms().num_errors == 1


def test_requiring_unique_mechanisms_accepts_a_unique_model() -> None:
    model = _model(DemError(probability=0.1, detectors=(0,), observables=(0,)))
    assert model.require_unique_mechanisms() is None


def test_requiring_unique_mechanisms_names_the_pair_and_the_operation() -> None:
    """The refusal has to say what is wrong and what resolves it."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
        DemError(probability=0.3, detectors=(0,), observables=(0,)),
    )
    with pytest.raises(ValueError) as refusal:
        model.require_unique_mechanisms()
    message = str(refusal.value)
    # The model stores its mechanisms sorted by signature and probability, so
    # the two members of the shared signature are the first and the second.
    assert "mechanisms 0 and 1" in message
    assert "D0 L0" in message
    assert "merge_duplicate_mechanisms()" in message


def test_merging_with_the_default_rule_combines_a_shared_signature() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0,), observables=(0,)),
        DemError(probability=0.05, detectors=(1,), observables=(0,)),
    )
    merged = model.merge_duplicate_mechanisms()
    assert merged.mechanisms_are_unique()
    assert merged.num_errors == 2
    assert [error.probability for error in merged.errors] == pytest.approx([0.26, 0.05])


def test_merging_is_idempotent() -> None:
    """Merging twice says nothing new, so a caller may merge defensively."""

    merged = _shared(0.1, 0.2, 0.05).merge_duplicate_mechanisms()
    assert merged.merge_duplicate_mechanisms() == merged


def test_merging_does_not_depend_on_the_order_the_mechanisms_arrived_in() -> None:
    """The constructor sorts, so how the caller assembled the model cannot show."""

    priors = [0.1, 0.2, 0.05, 0.3]
    forward = _shared(*priors).merge_duplicate_mechanisms()
    backward = _shared(*reversed(priors)).merge_duplicate_mechanisms()
    assert forward == backward


def test_merging_by_name_agrees_with_merging_by_member() -> None:
    model = _shared(0.1, 0.2)
    assert model.merge_duplicate_mechanisms(
        rule="clamped_linear_sum"
    ) == model.merge_duplicate_mechanisms(rule=DemMergeRule.CLAMPED_LINEAR_SUM)


def test_the_two_rules_disagree_so_the_argument_is_not_cosmetic() -> None:
    """The rules are not interchangeable, which is why the caller states one."""

    model = _shared(0.1, 0.2)
    parity = model.merge_duplicate_mechanisms()
    summed = model.merge_duplicate_mechanisms(rule=DemMergeRule.CLAMPED_LINEAR_SUM)
    assert parity != summed
    assert parity.errors[0].probability == pytest.approx(0.26)
    assert summed.errors[0].probability == pytest.approx(0.3)


def test_an_unknown_rule_is_refused_by_name() -> None:
    model = _shared(0.1)
    with pytest.raises(ValueError, match="'bayes' is not a valid DemMergeRule"):
        model.merge_duplicate_mechanisms(rule="bayes")  # type: ignore[arg-type]


def test_merging_a_unique_model_returns_an_equal_model() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0, 1), observables=()),
    )
    assert model.merge_duplicate_mechanisms() == model


def test_merging_an_empty_model_returns_an_equal_model() -> None:
    assert _model().merge_duplicate_mechanisms() == _model()


def test_a_lone_mechanism_keeps_its_probability_bit_for_bit() -> None:
    """A group of one is passed through and does not meet the arithmetic."""

    probability = 0.1234567890123456789
    model = _model(
        DemError(probability=probability, detectors=(0,), observables=(0,)),
        DemError(probability=0.1, detectors=(1,), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
    )
    merged = model.merge_duplicate_mechanisms()
    alone = next(error for error in merged.errors if error.detectors == (0,))
    assert alone.probability == probability
    assert repr(alone.probability) == repr(probability)


def test_merging_keeps_the_shape_and_every_signature() -> None:
    """It is a merge, not a filter: no fault is dropped by merging it."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0,), observables=(0,)),
        DemError(probability=0.3, detectors=(2,), observables=(0,)),
        detectors=3,
    )
    merged = model.merge_duplicate_mechanisms()
    assert (merged.num_detectors, merged.num_observables) == (
        model.num_detectors,
        model.num_observables,
    )
    assert {error.detectors for error in merged.errors} == {
        error.detectors for error in model.errors
    }
    assert merged.num_errors == 2


def test_merging_with_the_parity_rule_preserves_the_rates_exactly() -> None:
    """The merge is marginal-preserving, which is what makes it exact.

    A detector's rate is built from a product of ``1 - 2p`` factors over the
    mechanisms that touch it. A merged group's prior is the single ``p`` whose
    factor is that group's product, so grouping the factors cannot change the
    product, and the comparison below is exact rather than approximate.
    """

    model = _model(
        DemError(probability=0.01, detectors=(0,), observables=(0,)),
        DemError(probability=0.005, detectors=(0,), observables=(0,)),
        DemError(probability=0.005, detectors=(0,), observables=(0,)),
        DemError(probability=0.03, detectors=(1,), observables=(0,)),
        DemError(probability=0.02, detectors=(0, 1), observables=(0,)),
    )
    assert not model.mechanisms_are_unique()
    merged = model.merge_duplicate_mechanisms()
    assert bool((merged.detector_rates() == model.detector_rates()).all())
    assert bool((merged.observable_rates() == model.observable_rates()).all())


def test_the_sum_rule_does_not_preserve_the_rates() -> None:
    """Which is why the parity rule is the default and this one is stated."""

    model = _model(
        DemError(probability=0.01, detectors=(0,), observables=(0,)),
        DemError(probability=0.005, detectors=(0,), observables=(0,)),
        DemError(probability=0.005, detectors=(0,), observables=(0,)),
    )
    summed = model.merge_duplicate_mechanisms(rule=DemMergeRule.CLAMPED_LINEAR_SUM)
    assert not bool((summed.detector_rates() == model.detector_rates()).all())


def test_a_model_built_from_a_memory_circuit_is_already_unique() -> None:
    """Construction merges as it goes, so its own model never needs this call."""

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RepetitionCode(distance=3), rounds=3),
        noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01),
    )
    assert model.num_errors > 0
    assert model.mechanisms_are_unique()
    assert model.merge_duplicate_mechanisms() == model


def test_a_duplicated_mechanism_survives_a_text_round_trip_and_then_merges() -> None:
    """The text format allows a repeated mechanism, so the reader keeps it."""

    text = (
        "error(0.1) D0 L0\n"
        "error(0.2) D0 L0\n"
        "error(0.05) D1 L0\n"
        "detector D0\n"
        "detector D1\n"
        "logical_observable L0\n"
    )
    parsed = DetectorErrorModel.from_stim_text(text)
    assert parsed.num_errors == 3
    assert not parsed.mechanisms_are_unique()
    merged = parsed.merge_duplicate_mechanisms()
    assert merged.num_errors == 2
    assert DetectorErrorModel.from_stim_text(merged.to_stim_text()) == merged


def test_the_matcher_refuses_a_model_that_states_a_signature_twice() -> None:
    """A duplicate is a correctness hazard for a matcher, not a tidiness one.

    The graph keeps one edge per mechanism, so a signature stated twice becomes
    two parallel edges with two weights and the matcher would take the cheaper
    of them. Merging gives the fault the one weight its combined prior has.
    """

    model = _shared(0.1, 0.1)
    with pytest.raises(ValueError, match="merge_duplicate_mechanisms()"):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)

    merged = model.merge_duplicate_mechanisms()
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(merged)
    assert len(decoder.graph.edges) == 1
    assert decoder.graph.edges[0].probability == pytest.approx(_parity(0.1, 0.1))


def test_the_weight_a_duplicate_would_be_matched_at_is_not_a_faults_weight() -> None:
    """The refusal is about the number the matcher would use, not about tidiness.

    Two mechanisms of ``0.1`` and ``0.2`` on one signature are one fault of
    combined parity ``0.26``. Left apart, the graph holds two edges of weight
    ``log 9`` and ``log 4`` and the matcher charges the cheaper, ``log 4``, for
    a fault whose own weight is ``log(0.74 / 0.26)``. The charge is larger than
    the fault's, so a matcher would prefer a longer chain of other mechanisms
    over the mechanism that actually fired -- which is why the decoder refuses
    the model instead of weighting it.
    """

    model = _shared(0.1, 0.2)
    merged = model.merge_duplicate_mechanisms()
    merged_decoder = MinimumWeightMatchingDecoder.from_detector_error_model(merged)

    charged = merged_decoder.graph.edges[0].weight
    assert charged == pytest.approx(math.log(0.74 / 0.26))
    assert min(math.log(9.0), math.log(4.0)) > charged
    assert min(math.log(9.0), math.log(4.0)) - charged == pytest.approx(
        0.340325, abs=1e-6
    )


def test_a_fault_stated_twice_decodes_like_the_merged_fault() -> None:
    """One physical fault entered as two mechanisms decodes like the whole one.

    The two halves stand for the same location, so the model that states them
    separately has to be merged before a matcher may read it, and the merged
    model then predicts what the single unsplit mechanism predicts.
    """

    half = 0.02
    whole = _model(
        DemError(probability=_parity(half, half), detectors=(0, 1), observables=(0,)),
        DemError(probability=0.03, detectors=(1, 2), observables=(0,)),
    )
    split = _model(
        DemError(probability=half, detectors=(0, 1), observables=(0,)),
        DemError(probability=half, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.03, detectors=(1, 2), observables=(0,)),
    )
    assert not split.mechanisms_are_unique()
    merged = split.merge_duplicate_mechanisms()
    assert merged.mechanisms_are_unique()
    shared = next(error for error in merged.errors if error.detectors == (0, 1))
    assert shared.probability == pytest.approx(whole.errors[0].probability)

    merged_decoder = MinimumWeightMatchingDecoder.from_detector_error_model(merged)
    whole_decoder = MinimumWeightMatchingDecoder.from_detector_error_model(whole)
    for syndrome in [(), (0, 1), (1, 2), (0, 2)]:
        assert (
            merged_decoder.decode(syndrome).observables
            == whole_decoder.decode(syndrome).observables
        )
    # Neither model has a boundary edge, so a lone defect is refused by both
    # for the same reason rather than answered differently.
    with pytest.raises(CapabilityError, match="no chain of mechanisms"):
        merged_decoder.decode((0,))
    with pytest.raises(CapabilityError, match="no chain of mechanisms"):
        whole_decoder.decode((0,))
