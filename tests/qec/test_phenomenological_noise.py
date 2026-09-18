"""Unit coverage for the phenomenological noise record."""

from __future__ import annotations

import dataclasses

import pytest

from flagquantum.qec.noise import PhenomenologicalNoise, RepetitionNoiseProfile

pytestmark = pytest.mark.unit


def test_defaults_are_noiseless() -> None:
    noise = PhenomenologicalNoise()
    assert noise.data_flip == 0.0
    assert noise.measurement_flip == 0.0


def test_fields_round_trip() -> None:
    noise = PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.02)
    assert noise.data_flip == 0.05
    assert noise.measurement_flip == 0.02


def test_record_is_frozen() -> None:
    noise = PhenomenologicalNoise(data_flip=0.05)
    with pytest.raises(dataclasses.FrozenInstanceError):
        noise.data_flip = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("field", ["data_flip", "measurement_flip"])
def test_probability_bounds_are_enforced(field: str) -> None:
    with pytest.raises(ValueError):
        PhenomenologicalNoise(**{field: -0.01})
    with pytest.raises(ValueError):
        PhenomenologicalNoise(**{field: 1.01})


@pytest.mark.parametrize("field", ["data_flip", "measurement_flip"])
def test_bool_is_rejected_as_a_probability(field: str) -> None:
    with pytest.raises(TypeError):
        PhenomenologicalNoise(**{field: True})


def test_boundary_values_are_accepted() -> None:
    assert (
        PhenomenologicalNoise(data_flip=0.0, measurement_flip=1.0).measurement_flip
        == 1.0
    )


def test_frozen_profile_is_unchanged() -> None:
    """The additive rule: the frozen profile keeps its fields and defaults."""

    profile = RepetitionNoiseProfile()
    assert profile.data_bit_flip_probability == 0.0
    assert profile.syndrome_readout_error_probability == 0.0
    assert profile.final_readout_error_probability == 0.0
