"""Unit coverage for the detector-error-model records."""

from __future__ import annotations

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def test_error_stores_its_signature() -> None:
    error = DemError(probability=0.05, detectors=(0, 2), observables=(1,))
    assert error.probability == 0.05
    assert error.detectors == (0, 2)
    assert error.observables == (1,)


def test_error_normalizes_its_signature() -> None:
    error = DemError(probability=0.05, detectors=(2, 0, 2), observables=(1, 1))
    assert error.detectors == (0, 2)
    assert error.observables == (1,)


def test_error_rejects_an_empty_signature() -> None:
    with pytest.raises(ValueError, match="must flip at least one"):
        DemError(probability=0.05, detectors=(), observables=())


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_error_rejects_an_out_of_range_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        DemError(probability=probability, detectors=(0,), observables=())


def test_error_rejects_a_bool_probability() -> None:
    with pytest.raises(TypeError, match="must be a real probability"):
        DemError(probability=True, detectors=(0,), observables=())


def test_error_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="non-negative indices"):
        DemError(probability=0.05, detectors=(-1,), observables=())


def test_model_stores_its_shape() -> None:
    model = DetectorErrorModel(
        num_detectors=4,
        num_observables=1,
        errors=(DemError(probability=0.05, detectors=(0,), observables=(0,)),),
    )
    assert model.num_detectors == 4
    assert model.num_observables == 1
    assert model.num_errors == 1


def test_model_accepts_no_errors() -> None:
    model = DetectorErrorModel(num_detectors=2, num_observables=1, errors=())
    assert model.num_errors == 0


def test_model_rejects_an_out_of_range_detector() -> None:
    with pytest.raises(ValueError, match="detector index"):
        DetectorErrorModel(
            num_detectors=2,
            num_observables=1,
            errors=(DemError(probability=0.05, detectors=(2,), observables=()),),
        )


def test_model_rejects_an_out_of_range_observable() -> None:
    with pytest.raises(ValueError, match="observable index"):
        DetectorErrorModel(
            num_detectors=2,
            num_observables=1,
            errors=(DemError(probability=0.05, detectors=(0,), observables=(1,)),),
        )


def test_model_orders_errors_canonically() -> None:
    """Equal models built in different orders compare equal."""

    first = DemError(probability=0.05, detectors=(1,), observables=())
    second = DemError(probability=0.02, detectors=(0,), observables=(0,))
    forward = DetectorErrorModel(
        num_detectors=2, num_observables=1, errors=(first, second)
    )
    backward = DetectorErrorModel(
        num_detectors=2, num_observables=1, errors=(second, first)
    )
    assert forward == backward
    assert forward.errors == (second, first)


def test_model_requires_a_positive_detector_count() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel(num_detectors=0, num_observables=1, errors=())


def test_model_allows_a_model_without_observables() -> None:
    model = DetectorErrorModel(num_detectors=1, num_observables=0, errors=())
    assert model.num_observables == 0
