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
    """Row is the detector, column the error, for every index in the model.

    Detector index 2 exceeds the column count, so an implementation that wrote
    ``matrix[column, index]`` would raise ``IndexError`` rather than happen to
    land on the same cells.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0, 2), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
    )
    matrix = model.detector_error_matrix()
    assert matrix.shape == (3, 2)
    assert matrix.dtype == torch.int8
    assert matrix.tolist() == [[1, 0], [0, 1], [1, 0]]


def test_observable_matrix_orientation_and_values() -> None:
    """Row is the observable, column the error, for every index in the model."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(2,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
        observables=3,
    )
    matrix = model.observables_flips_matrix()
    assert matrix.shape == (3, 2)
    assert matrix.dtype == torch.int8
    assert matrix.tolist() == [[0, 0], [0, 0], [1, 0]]


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


def test_observable_index_selects_its_own_row_and_rate() -> None:
    """Each observable index is addressed on its own, not merely "some" one.

    With two logical observables an implementation that collapsed the axis to
    index zero would put both entries and both rates in slot zero.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0,), observables=(1,)),
        detectors=1,
        observables=2,
    )
    assert model.observables_flips_matrix().tolist() == [[1, 0], [0, 1]]
    assert model.observable_rates().tolist() == pytest.approx([0.1, 0.2])


def test_observable_rates_are_empty_without_observables() -> None:
    """An observable-free model returns a typed empty tensor."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        detectors=1,
        observables=0,
    )
    rates = model.observable_rates()
    assert rates.shape == (0,)
    assert rates.dtype == torch.float64
    assert model.observables_flips_matrix().shape == (0, 1)


# ---------------------------------------------------------------------------
# the parallel rate column
# ---------------------------------------------------------------------------


def test_error_rates_are_the_mechanisms_own_in_column_order() -> None:
    """Entry ``e`` is column ``e``'s rate, read against the column's own support.

    The mechanisms are stated out of column order, have distinct rates *and*
    distinct supports, and their detector order is the reverse of their rate
    order, so an implementation that returned the rates as given, sorted by rate,
    or in the order of the first detector each one touches would land on a
    different vector here. Each entry is checked beside the column it claims to
    weight rather than on its own.
    """

    model = _model(
        DemError(probability=0.3, detectors=(2,), observables=()),
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
    )
    matrix = model.detector_error_matrix()
    rates = model.error_rates
    assert rates == (0.1, 0.2, 0.3)
    assert [error.probability for error in model.errors] == [0.1, 0.2, 0.3]
    for column, rate in enumerate(rates):
        support = matrix[:, column].nonzero().flatten().tolist()
        assert support == list(model.errors[column].detectors)
        assert rate == model.errors[column].probability


def test_the_column_order_is_the_models_own_not_the_callers() -> None:
    """The pairing is with the matrices, so the vector follows the column order.

    A model normalizes its mechanisms into one sequence and every column view is
    taken from that sequence; a caller that stated its mechanisms in the order it
    thought about them would mis-weight every column if this vector kept the
    caller's order while the matrices did not.
    """

    forwards = _model(
        DemError(probability=0.4, detectors=(0,), observables=()),
        DemError(probability=0.6, detectors=(1,), observables=()),
        detectors=2,
    )
    backwards = _model(
        DemError(probability=0.6, detectors=(1,), observables=()),
        DemError(probability=0.4, detectors=(0,), observables=()),
        detectors=2,
    )
    assert forwards.error_rates == backwards.error_rates == (0.4, 0.6)
    assert torch.equal(
        forwards.detector_error_matrix(), backwards.detector_error_matrix()
    )


def test_error_rates_column_count_is_num_errors() -> None:
    """The vector indexes the matrices' columns, so its length is their width."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        detectors=2,
    )
    assert len(model.error_rates) == model.num_errors
    assert len(model.error_rates) == model.detector_error_matrix().shape[1]
    assert len(model.error_rates) == model.observables_flips_matrix().shape[1]


def test_error_rates_are_not_the_detector_marginals() -> None:
    """The two quantities differ exactly where a detector has two mechanisms.

    A single error is its own marginal, so the two agree at column 1 and diverge
    at column 0. An accessor that returned ``detector_rates()`` here, or that
    composed two mechanisms on one detector by parity, would pass a one-mechanism
    check and fail this one.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(0,), observables=()),
        detectors=2,
    )
    assert model.error_rates == (0.1, 0.2)
    assert model.detector_rates().tolist() == pytest.approx([0.26, 0.0])
    assert model.error_rates[1] == pytest.approx(model.detector_rates()[1].item() + 0.2)


def test_error_rates_of_an_empty_model_are_an_empty_tuple() -> None:
    model = DetectorErrorModel(num_detectors=3, num_observables=2, errors=())
    assert model.error_rates == ()
    assert model.num_errors == 0


def test_a_grouped_model_keeps_each_members_own_rate() -> None:
    """Exclusivity says which columns are alternatives, not what they are worth.

    Reading a group as a distribution over its members would put the summed mass
    here; the vector states what each mechanism itself costs, and
    :meth:`detector_rates` stays the place the group is folded.
    """

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(), error_id=0),
        DemError(probability=0.2, detectors=(1,), observables=(), error_id=0),
        detectors=2,
    )
    assert model.error_rates == (0.1, 0.2)
    assert model.error_ids == (0, 0)
    assert model.exclusive_groups() == ((0, 1),)


def test_error_rates_is_a_tuple_of_plain_floats() -> None:
    """The vector is a stated tuple, not the marginal tensors under a new name."""

    model = _model(
        DemError(probability=0.25, detectors=(0,), observables=()),
        detectors=1,
    )
    rates = model.error_rates
    assert isinstance(rates, tuple)
    assert all(isinstance(rate, float) for rate in rates)
    assert rates == (0.25,)
