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


def test_an_error_free_model_never_flips() -> None:
    model = DetectorErrorModel(num_detectors=3, num_observables=1, errors=())
    sample = model.dem_sampling(shots=8, seed=0)
    assert sample.detectors.shape == (8, 3)
    assert int(sample.detectors.sum()) == 0
    assert int(sample.observables.sum()) == 0


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
