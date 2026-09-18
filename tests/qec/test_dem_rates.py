"""Unit coverage for parity matrices and exact marginal rates."""

from __future__ import annotations

import pytest
import torch

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_detector_matrix_orientation_and_values() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
    )
    matrix = model.detector_error_matrix()
    assert matrix.shape == (3, 2)
    assert matrix.dtype == torch.int8
    assert matrix.tolist() == [[1, 0], [1, 1], [0, 0]]


def test_observable_matrix_orientation_and_values() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
    )
    matrix = model.observables_flips_matrix()
    assert matrix.shape == (1, 2)
    assert matrix.dtype == torch.int8
    assert matrix.tolist() == [[1, 0]]


def test_empty_model_has_empty_matrices() -> None:
    model = DetectorErrorModel(num_detectors=3, num_observables=2, errors=())
    assert model.detector_error_matrix().shape == (3, 0)
    assert model.observables_flips_matrix().shape == (2, 0)


def test_single_error_rates_are_its_own_probabilities() -> None:
    model = _model(
        DemError(probability=0.25, detectors=(0, 1), observables=(0,)),
        detectors=2,
    )
    assert model.detector_rates().tolist() == pytest.approx([0.25, 0.25])
    assert model.observable_rates().tolist() == pytest.approx([0.25])


def test_untouched_detector_has_zero_rate() -> None:
    model = _model(
        DemError(probability=0.3, detectors=(0,), observables=()), detectors=2
    )
    assert model.detector_rates().tolist() == pytest.approx([0.3, 0.0])


def test_shared_detector_combines_by_parity_not_by_sum() -> None:
    """Two mechanisms on one detector flip it when exactly one fires."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(0,), observables=()),
        detectors=1,
    )
    expected = 0.1 * 0.8 + 0.2 * 0.9
    assert model.detector_rates().tolist() == pytest.approx([expected])


def test_three_mechanisms_use_the_closed_form() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(0,), observables=()),
        DemError(probability=0.4, detectors=(0,), observables=()),
        detectors=1,
    )
    expected = (1 - (0.8 * 0.6 * 0.2)) / 2
    assert model.detector_rates().tolist() == pytest.approx([expected])


def test_rates_are_independent_across_detectors() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.2, detectors=(1, 2), observables=(0,)),
        detectors=3,
    )
    rates = model.detector_rates().tolist()
    assert rates[0] == pytest.approx(0.1)
    assert rates[1] == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)
    assert rates[2] == pytest.approx(0.2)


def test_a_certain_error_yields_certain_rates() -> None:
    model = _model(
        DemError(probability=1.0, detectors=(0,), observables=(0,)), detectors=1
    )
    assert model.detector_rates().tolist() == pytest.approx([1.0])
    assert model.observable_rates().tolist() == pytest.approx([1.0])


def test_rates_are_float64() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()), detectors=1
    )
    assert model.detector_rates().dtype == torch.float64
    assert model.observable_rates().dtype == torch.float64


def test_observable_rate_ignores_detector_only_errors() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
        detectors=2,
    )
    assert model.observable_rates().tolist() == pytest.approx([0.2])


def test_observable_rates_are_empty_without_observables() -> None:
    """A model may declare no observable, and the empty early return is typed."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        detectors=1,
        observables=0,
    )
    rates = model.observable_rates()
    assert rates.shape == (0,)
    assert rates.dtype == torch.float64
