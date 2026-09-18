"""Unit coverage for seeded detector error model sampling."""

from __future__ import annotations

import pytest
import torch

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_sample_shapes_and_dtypes() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)), detectors=4
    )
    sample = model.dem_sampling(shots=16, seed=0)
    assert sample.detectors.shape == (16, 4)
    assert sample.observables.shape == (16, 1)
    assert sample.detectors.dtype == torch.int8
    assert sample.observables.dtype == torch.int8
    assert sample.shots == 16


def test_sampling_is_reproducible_for_a_seed() -> None:
    """A seed reproduces its stream, and it is the seed that selects it.

    The second half is not redundant: ``torch.Generator()`` starts every new
    generator at the same fixed default state, so two calls that both ignore
    ``seed`` still agree with each other. Only comparing a seeded draw against
    an unseeded one shows that the seed was read at all.
    """

    model = _model(
        DemError(probability=0.3, detectors=(0,), observables=(0,)), detectors=2
    )
    first = model.dem_sampling(shots=64, seed=7)
    second = model.dem_sampling(shots=64, seed=7)
    assert torch.equal(first.detectors, second.detectors)
    assert torch.equal(first.observables, second.observables)
    unseeded = model.dem_sampling(shots=64)
    assert not torch.equal(first.detectors, unseeded.detectors)
    assert not torch.equal(first.observables, unseeded.observables)


def test_different_seeds_differ() -> None:
    model = _model(
        DemError(probability=0.3, detectors=(0,), observables=(0,)), detectors=2
    )
    assert not torch.equal(
        model.dem_sampling(shots=64, seed=1).detectors,
        model.dem_sampling(shots=64, seed=2).detectors,
    )


def test_samples_compare_by_tensor_content() -> None:
    """Two samples with the same content are equal, and the record is unhashable.

    The comparison has to be by value: an equality that fell through to
    ``bool(tensor)`` would raise ``RuntimeError`` on any multi-element field
    instead of answering, and a hash that disagreed with equality would be
    worse than no hash at all.
    """

    model = _model(
        DemError(probability=0.5, detectors=(0,), observables=(0,)), detectors=2
    )
    first = model.dem_sampling(shots=32, seed=5)
    same = model.dem_sampling(shots=32, seed=5)
    other = model.dem_sampling(shots=32, seed=6)
    assert first == same
    assert not (first == other)
    assert first != other
    assert (first == "not a sample") is False
    with pytest.raises(TypeError, match="unhashable type"):
        hash(first)


def test_an_error_free_model_never_flips() -> None:
    model = DetectorErrorModel(num_detectors=3, num_observables=1, errors=())
    sample = model.dem_sampling(shots=8, seed=0)
    assert sample.detectors.shape == (8, 3)
    assert int(sample.detectors.sum()) == 0
    assert int(sample.observables.sum()) == 0


def test_samples_that_differ_only_in_their_observables_are_not_equal() -> None:
    """Equality reads the observable column, not only the detector column.

    A model whose only mechanism flips the observable leaves every sample's
    detector column all-zero, so two different draws differ in ``observables``
    alone. ``test_samples_compare_by_tensor_content`` cannot show this: its
    model flips detector and observable together, so an equality that dropped
    the observables conjunct answers the same for every sample it builds.
    """

    model = _model(
        DemError(probability=0.5, observables=(0,)), detectors=2, observables=1
    )
    first = model.dem_sampling(shots=32, seed=5)
    other = model.dem_sampling(shots=32, seed=6)
    assert torch.equal(first.detectors, other.detectors)
    assert not torch.equal(first.observables, other.observables)
    assert not (first == other)
    assert first != other


def test_samples_that_differ_only_in_their_detectors_are_not_equal() -> None:
    """The mirror case: an equality that read only ``observables`` also fails here."""

    model = _model(
        DemError(probability=0.5, detectors=(0,), observables=()), detectors=2
    )
    first = model.dem_sampling(shots=32, seed=5)
    other = model.dem_sampling(shots=32, seed=6)
    assert not torch.equal(first.detectors, other.detectors)
    assert torch.equal(first.observables, other.observables)
    assert not (first == other)
    assert first != other


def test_a_certain_error_always_flips() -> None:
    model = _model(
        DemError(probability=1.0, detectors=(0,), observables=(0,)), detectors=2
    )
    sample = model.dem_sampling(shots=8, seed=0)
    assert int(sample.detectors[:, 0].sum()) == 8
    assert int(sample.observables[:, 0].sum()) == 8


def test_a_shared_error_flips_both_detectors_together() -> None:
    """One mechanism that flips two detectors flips them together.

    The two columns have the *same* marginal rate, so an implementation that
    drew an independent Bernoulli per detector would satisfy every rate
    comparison and still split the pair here.  The count assertion keeps the
    equality from holding vacuously: an all-zero sample would make the columns
    equal without any mechanism ever firing.
    """

    model = _model(
        DemError(probability=0.5, detectors=(0, 1), observables=()), detectors=2
    )
    sample = model.dem_sampling(shots=256, seed=3)
    assert torch.equal(sample.detectors[:, 0], sample.detectors[:, 1])
    assert 0 < int(sample.detectors[:, 0].sum()) < 256


def test_sampled_rates_track_the_exact_rates() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0, 1), observables=(0,)),
        detectors=2,
    )
    sample = model.dem_sampling(shots=20_000, seed=11)
    sampled = sample.detectors.to(torch.float64).mean(dim=0)
    assert sampled.tolist() == pytest.approx(model.detector_rates().tolist(), abs=0.01)
    sampled_observable = sample.observables.to(torch.float64).mean(dim=0)
    assert sampled_observable.tolist() == pytest.approx(
        model.observable_rates().tolist(), abs=0.01
    )


@pytest.mark.parametrize("shots", [0, -1])
def test_shots_must_be_positive(shots: int) -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()), detectors=1
    )
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        model.dem_sampling(shots=shots, seed=0)


def test_bool_shots_are_rejected() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()), detectors=1
    )
    with pytest.raises(TypeError, match="shots must be a positive integer"):
        model.dem_sampling(shots=True, seed=0)


def test_shots_must_be_keyword_only() -> None:
    """``shots`` is keyword-only, so a positional call does not reach the guard."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()), detectors=1
    )
    with pytest.raises(TypeError, match="positional argument"):
        model.dem_sampling(8)


@pytest.mark.parametrize("seed", [True, 1.5])
def test_seed_must_be_an_integer_or_none(seed: object) -> None:
    """A ``bool`` seed is a rejected seed here, not a value of one."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()), detectors=1
    )
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        model.dem_sampling(shots=4, seed=seed)


def test_a_model_without_observables_samples_an_empty_column() -> None:
    """``num_observables == 0`` yields a ``(shots, 0)`` column, not a 1-wide one."""

    model = _model(
        DemError(probability=1.0, detectors=(0, 1)), detectors=2, observables=0
    )
    sample = model.dem_sampling(shots=8, seed=0)
    assert sample.detectors.shape == (8, 2)
    assert sample.observables.shape == (8, 0)
    assert sample.observables.dtype == torch.int8
    assert sample.shots == 8
    assert int(sample.detectors.sum()) == 16
