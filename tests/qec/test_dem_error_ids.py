"""Unit coverage for correlated mechanisms stated as error ids.

An error id is the one statement about a detector error model that its parity
matrices cannot carry: two columns of a parity matrix are independent by
construction, so a model that has to say that two mechanisms are alternatives
has nowhere else to put it. These tests pin the statement, its arithmetic, and
the three places that refuse it rather than quietly dropping it.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec import MinimumWeightMatchingDecoder
from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 2):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


# --------------------------------------------------------------------------
# the statement itself
# --------------------------------------------------------------------------


def test_a_mechanism_has_no_id_by_default() -> None:
    error = DemError(probability=0.1, detectors=(0, 1), observables=(0,))
    assert error.error_id is None


def test_an_id_must_be_a_non_negative_integer() -> None:
    """The id is a label, so it is an index and not a probability or a flag."""

    with pytest.raises(TypeError, match="error id must be an integer"):
        DemError(probability=0.1, detectors=(0,), error_id=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error id must be non-negative"):
        DemError(probability=0.1, detectors=(0,), error_id=-1)
    # A bool is an int in Python, and True would be the id 1 rather than a flag.
    with pytest.raises(TypeError, match="error id must be an integer"):
        DemError(probability=0.1, detectors=(0,), error_id=True)
    assert DemError(probability=0.1, detectors=(0,), error_id=0).error_id == 0


def test_an_id_is_part_of_the_record_so_it_takes_part_in_equality() -> None:
    plain = DemError(probability=0.1, detectors=(0,))
    identified = DemError(probability=0.1, detectors=(0,), error_id=4)
    assert plain != identified
    assert plain == DemError(probability=0.1, detectors=(0,))


def test_mechanisms_with_the_same_signature_may_be_alternatives() -> None:
    """One fault with two possible causes is one signature and two mechanisms.

    This is the case the parity matrices cannot hold, so it is also the case that
    ``mechanisms_are_unique`` reports and that no decoder may read without
    folding.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0, 1), error_id=0),
        DemError(probability=0.2, detectors=(0, 1), error_id=0),
    )
    assert model.num_errors == 2
    assert not model.mechanisms_are_unique()


def test_an_id_free_model_states_nothing() -> None:
    model = _model(DemError(probability=0.1, detectors=(0,)), detectors=2)
    assert model.error_ids is None
    assert model.stated_error_ids() == ()
    assert model.exclusive_groups() == ()


def test_the_projected_vector_is_parallel_to_the_mechanisms() -> None:
    """An unstated id is the id of a group of one, numbered above the stated ids.

    The model stores its mechanisms sorted, so the vector is parallel to that
    sorted order and not to the order the mechanisms were passed in: the two
    members of id 7 are stored first because their signatures sort first, and the
    unstated mechanism sits last with the id that follows the stated ones.
    """

    model = _model(
        DemError(probability=0.1, detectors=(1, 2), error_id=7),
        DemError(probability=0.2, detectors=(0, 1), error_id=7),
        DemError(probability=0.3, detectors=(3,)),
    )
    ids = model.error_ids
    assert ids == (7, 7, 8)
    assert len(ids) == model.num_errors
    # The vector induces a partition, and the lone mechanism is a group of one.
    assert model.exclusive_groups() == ((0, 1),)
    assert model.stated_error_ids() == (7,)


def test_renumbering_a_group_leaves_its_rates_unchanged() -> None:
    """Ids are opaque labels, so what the model says is the partition."""

    errors = (
        DemError(probability=0.1, detectors=(0,), error_id=0),
        DemError(probability=0.2, detectors=(1,), error_id=0),
    )
    other = (
        DemError(probability=0.1, detectors=(0,), error_id=41),
        DemError(probability=0.2, detectors=(1,), error_id=41),
    )
    first = _model(*errors, detectors=2)
    second = _model(*other, detectors=2)
    assert torch.equal(first.detector_rates(), second.detector_rates())
    assert first.error_ids != second.error_ids


def test_a_mechanism_may_be_in_at_most_one_group() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), error_id=0),
        DemError(probability=0.1, detectors=(1,), error_id=1),
        DemError(probability=0.1, detectors=(2,), error_id=0),
    )
    assert model.exclusive_groups() == ((0, 2),)
    assert model.error_ids == (0, 1, 0)


# --------------------------------------------------------------------------
# the arithmetic of a group
# --------------------------------------------------------------------------


def test_a_group_adds_where_independent_mechanisms_take_parity() -> None:
    """At most one member fires, so the members touching a detector are disjoint.

    The same two probabilities give 0.26 when the mechanisms are independent and
    0.3 when they exclude each other, and the pair of assertions is what shows the
    id changed the arithmetic rather than the record alone.
    """

    independent = _model(
        DemError(probability=0.1, detectors=(0,)),
        DemError(probability=0.2, detectors=(0,)),
        detectors=1,
    )
    exclusive = _model(
        DemError(probability=0.1, detectors=(0,), error_id=0),
        DemError(probability=0.2, detectors=(0,), error_id=0),
        detectors=1,
    )
    assert independent.detector_rates().tolist() == pytest.approx([0.26])
    assert exclusive.detector_rates().tolist() == pytest.approx([0.3])


def test_a_group_composes_by_parity_with_the_faults_outside_it() -> None:
    """A group is one fault to the rest of the model, not several."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), error_id=3),
        DemError(probability=0.2, detectors=(0,), error_id=3),
        DemError(probability=0.25, detectors=(0,)),
        detectors=1,
    )
    # The group flips detector 0 with probability 0.3, and the lone mechanism is
    # independent of it, so the detector's rate is the parity of 0.3 and 0.25.
    expected = 0.3 * 0.75 + 0.25 * 0.7
    assert model.detector_rates().tolist() == pytest.approx([expected])


def test_a_group_that_does_not_touch_a_detector_leaves_its_rate_alone() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), error_id=0),
        DemError(probability=0.2, detectors=(1,), error_id=0),
        detectors=3,
    )
    assert model.detector_rates().tolist() == pytest.approx([0.1, 0.2, 0.0])


def test_observable_rates_honour_a_group_too() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(1,), error_id=5),
        DemError(probability=0.4, detectors=(1,), observables=(1,), error_id=5),
        detectors=2,
        observables=2,
    )
    assert model.observable_rates().tolist() == pytest.approx([0.0, 0.5])


def test_a_group_may_hold_more_than_two_members() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), error_id=2),
        DemError(probability=0.2, detectors=(0,), error_id=2),
        DemError(probability=0.3, detectors=(0,), error_id=2),
        detectors=1,
    )
    assert model.detector_rates().tolist() == pytest.approx([0.6])


def test_a_group_that_fires_every_shot_is_admitted() -> None:
    """Summing to exactly one is representable: nothing is left for no member.

    The rate is compared against ``0.0`` as well, because the complement of a
    group that always fires is nothing and a rate of exactly one is the boundary
    the parity fold cannot reach on its own.
    """

    model = _model(
        DemError(probability=0.4, detectors=(0,), observables=(0,), error_id=0),
        DemError(probability=0.6, detectors=(1,), observables=(0,), error_id=0),
        detectors=2,
        observables=1,
    )
    assert model.detector_rates().tolist() == pytest.approx([0.4, 0.6])
    assert model.observable_rates().tolist() == pytest.approx([1.0])


def test_a_group_that_overflows_every_shot_is_refused() -> None:
    """Renormalizing would change every rate the caller read, so it refuses."""

    with pytest.raises(ValueError) as refusal:
        _model(
            DemError(probability=0.7, detectors=(0,), error_id=9),
            DemError(probability=0.7, detectors=(1,), error_id=9),
            detectors=2,
        )
    message = str(refusal.value)
    assert "9" in message
    assert "1.4" in message
    assert "at most one shot's worth of probability" in message


def test_the_overflow_bound_is_per_group_and_not_per_model() -> None:
    """Two groups may each hold half a shot, which states no conflict."""

    model = _model(
        DemError(probability=0.5, detectors=(0,), error_id=0),
        DemError(probability=0.5, detectors=(1,), error_id=0),
        DemError(probability=0.5, detectors=(2,), error_id=1),
        DemError(probability=0.5, detectors=(3,), error_id=1),
        detectors=4,
    )
    assert model.detector_rates().tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def _rate_tolerance(shots: int) -> float:
    """Six standard errors of a Bernoulli rate, which is at most 3 / sqrt(n).

    A rate's standard error is ``sqrt(p (1 - p) / n)``, whose largest value over
    ``p`` is ``0.5 / sqrt(n)``, so six of them is bounded by ``3 / sqrt(n)``
    without knowing the rate. The bound is stated rather than tuned: a sampled
    rate outside it is a disagreement rather than a draw.
    """

    return 3.0 / math.sqrt(shots)


def test_sampled_rates_match_the_exact_rates_for_a_grouped_model() -> None:
    """The two computations of one quantity are independent enough to compare.

    ``detector_rates`` folds the model's arithmetic in closed form and the sample
    comes from the generator, so agreement is evidence about the arithmetic and
    about the sampler at once.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,), error_id=0),
        DemError(probability=0.2, detectors=(0,), observables=(0,), error_id=0),
        DemError(probability=0.05, detectors=(1,), observables=(1,)),
        detectors=2,
        observables=2,
    )
    shots = 200_000
    sample = model.dem_sampling(shots=shots, seed=20_260_111)
    tolerance = _rate_tolerance(shots)
    sampled = sample.detectors.double().mean(dim=0).tolist()
    assert sampled == pytest.approx(model.detector_rates().tolist(), abs=tolerance)
    sampled = sample.observables.double().mean(dim=0).tolist()
    assert sampled == pytest.approx(model.observable_rates().tolist(), abs=tolerance)


def test_no_shot_fires_two_members_of_one_group() -> None:
    """The exclusion is asserted directly, not only through the marginal rates.

    A sampler that drew the members independently would reproduce neither the
    grouped rate nor this count, and the two members here are given signatures
    that overlap on one detector so a double firing is visible as a bit rather
    than only as a rate.
    """

    model = _model(
        DemError(probability=0.4, detectors=(0,), observables=(0,), error_id=0),
        DemError(probability=0.4, detectors=(0,), observables=(1,), error_id=0),
        detectors=1,
        observables=2,
    )
    sample = model.dem_sampling(shots=50_000, seed=11)
    both = (sample.observables[:, 0] == 1) & (sample.observables[:, 1] == 1)
    assert int(both.sum()) == 0
    # And the group really is firing, so the count above is a fact about the
    # exclusion rather than about a group that never fires at all.
    assert int(sample.detectors[:, 0].sum()) > 0.7 * 50_000


def test_a_member_keeps_the_probability_it_states() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,), error_id=0),
        DemError(probability=0.3, detectors=(0,), observables=(1,), error_id=0),
        detectors=1,
        observables=2,
    )
    shots = 200_000
    sample = model.dem_sampling(shots=shots, seed=5)
    tolerance = _rate_tolerance(shots)
    assert float(sample.observables[:, 0].double().mean()) == pytest.approx(
        0.1, abs=tolerance
    )
    assert float(sample.observables[:, 1].double().mean()) == pytest.approx(
        0.3, abs=tolerance
    )


def test_sampling_a_grouped_model_is_reproducible_for_a_seed() -> None:
    model = _model(
        DemError(probability=0.2, detectors=(0,), error_id=0),
        DemError(probability=0.2, detectors=(1,), error_id=0),
        detectors=2,
    )
    assert model.dem_sampling(shots=64, seed=3) == model.dem_sampling(shots=64, seed=3)
    assert model.dem_sampling(shots=64, seed=3) != model.dem_sampling(shots=64, seed=4)


def test_a_lone_id_does_not_group_anything() -> None:
    """A mechanism with an id no other mechanism holds is on its own.

    Its rates and its group list are the ones an id-free model would report,
    which is what makes an id safe to set without meaning to exclude anything.
    """

    lone = _model(
        DemError(probability=0.1, detectors=(0,), error_id=0),
        DemError(probability=0.2, detectors=(1,)),
        detectors=2,
    )
    plain = _model(
        DemError(probability=0.1, detectors=(0,)),
        DemError(probability=0.2, detectors=(1,)),
        detectors=2,
    )
    assert lone.exclusive_groups() == ()
    assert torch.equal(lone.detector_rates(), plain.detector_rates())


# --------------------------------------------------------------------------
# the three refusals
# --------------------------------------------------------------------------


def test_stim_text_refuses_a_correlated_model_and_names_the_ids() -> None:
    """The format reads every error instruction as independent, so printing lies."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), error_id=4),
        DemError(probability=0.2, detectors=(1,), error_id=4),
        detectors=2,
    )
    with pytest.raises(ValueError) as refusal:
        model.to_stim_text()
    message = str(refusal.value)
    assert "4" in message
    assert "independent mechanism" in message


def test_stim_text_still_prints_a_model_with_no_ids() -> None:
    model = _model(DemError(probability=0.1, detectors=(0, 1)), detectors=2)
    text = model.to_stim_text()
    assert text.startswith("error(0.1) D0 D1\n")
    assert DetectorErrorModel.from_stim_text(text) == model


def test_a_text_never_reads_back_an_id() -> None:
    """Upstream states the ids a text is read into are always empty; so are these."""

    text = "error(0.1) D0 D1\ndetector D0\ndetector D1\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.error_ids is None
    assert model.stated_error_ids() == ()


def test_merging_refuses_a_correlated_model_and_names_the_ids() -> None:
    """Either rule would invent a shot in which two alternatives both fired."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), error_id=2),
        DemError(probability=0.2, detectors=(0,), error_id=2),
        detectors=1,
    )
    for rule in ("independent_parity", "clamped_linear_sum"):
        with pytest.raises(ValueError) as refusal:
            model.merge_duplicate_mechanisms(rule=rule)
        message = str(refusal.value)
        assert "2" in message
        assert "no merge under either rule" in message


def test_merging_a_model_with_no_ids_is_unchanged() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,)),
        DemError(probability=0.2, detectors=(0,)),
        detectors=1,
    )
    merged = model.merge_duplicate_mechanisms()
    assert merged.num_errors == 1
    assert merged.error_ids is None
    assert merged.detector_rates().tolist() == pytest.approx(
        model.detector_rates().tolist()
    )


def test_the_matching_decoder_refuses_a_correlated_model() -> None:
    """One weight per mechanism is the weight of a fault that fires alone."""

    model = _model(
        DemError(probability=0.1, detectors=(0, 1), error_id=8),
        DemError(probability=0.2, detectors=(1, 2), error_id=8),
        detectors=3,
    )
    with pytest.raises(CapabilityError) as refusal:
        MinimumWeightMatchingDecoder.from_detector_error_model(model)
    message = str(refusal.value)
    assert "alternatives" in message
    assert "folded into a single mechanism" in message


def test_the_matching_decoder_still_takes_a_model_with_no_ids() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        detectors=2,
        observables=1,
    )
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    assert decoder.decode((0, 1)).observables == (0,)


def test_an_id_free_model_reports_an_empty_group_list_before_the_refusals() -> None:
    """The refusals read ``stated_error_ids``, so the empty case is not a branch.

    An id-free model takes every one of the three routes above unchanged, which
    is the property the field's arrival had to preserve.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        detectors=2,
        observables=1,
    )
    assert model.stated_error_ids() == ()
    assert model.to_stim_text().startswith("error(0.1) D0 D1 L0\n")
    assert model.merge_duplicate_mechanisms() == model
    MinimumWeightMatchingDecoder.from_detector_error_model(model)
