"""Unit coverage for the stim text interchange."""

from __future__ import annotations

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 3, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_exact_text_for_a_hand_built_model() -> None:
    model = _model(
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.125, detectors=(2,), observables=()),
    )
    assert model.to_stim_text() == (
        "error(0.05) D0 D1 L0\n"
        "error(0.125) D2\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
    )


def test_error_free_model_still_declares_its_shape() -> None:
    model = DetectorErrorModel(num_detectors=2, num_observables=1, errors=())
    assert model.to_stim_text() == ("detector D0\ndetector D1\nlogical_observable L0\n")


def test_round_trip_preserves_the_model() -> None:
    model = _model(
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        DemError(probability=1 / 3, detectors=(2,), observables=(0,)),
        detectors=3,
    )
    assert DetectorErrorModel.from_stim_text(model.to_stim_text()) == model


def test_round_trip_preserves_a_shape_no_error_touches() -> None:
    """The declaration lines are what make this lossless."""

    model = DetectorErrorModel(num_detectors=5, num_observables=2, errors=())
    restored = DetectorErrorModel.from_stim_text(model.to_stim_text())
    assert restored.num_detectors == 5
    assert restored.num_observables == 2


def test_round_trip_is_stable() -> None:
    model = _model(
        DemError(probability=1 / 3, detectors=(0,), observables=(0,)), detectors=1
    )
    once = model.to_stim_text()
    assert DetectorErrorModel.from_stim_text(once).to_stim_text() == once


def test_parses_coordinate_declarations() -> None:
    text = (
        "error(0.1) D0 D1 L0\n"
        "error(0.2) D2\n"
        "detector(1, 2) D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3
    assert model.num_observables == 1
    assert model.num_errors == 2


def test_rejects_an_unsupported_instruction() -> None:
    with pytest.raises(ValueError, match="shift_detectors"):
        DetectorErrorModel.from_stim_text("shift_detectors 1\n")


def test_rejects_a_repeat_block() -> None:
    with pytest.raises(ValueError, match="repeat"):
        DetectorErrorModel.from_stim_text("repeat 2 {\n error(0.1) D0\n}\n")


def test_rejects_an_error_with_no_effect() -> None:
    with pytest.raises(ValueError, match="must flip at least one"):
        DetectorErrorModel.from_stim_text("error(0.1)\n")


def test_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        DetectorErrorModel.from_stim_text("error(1.5) D0\n")


def test_rejects_a_malformed_target() -> None:
    with pytest.raises(ValueError, match="D or L"):
        DetectorErrorModel.from_stim_text("error(0.1) X0\n")


def test_rejects_a_model_with_no_detectors() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text("logical_observable L0\n")


def test_rejects_a_bare_string_without_declarations() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text("error(0.1) D0\n")


def test_parses_the_error_content_not_only_the_shape() -> None:
    """A parser that dropped a target would satisfy a shape-only check."""

    text = (
        "error(0.1) D0 D2 L0 L1\n"
        "error(0.2) D1\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
        "logical_observable L1\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.errors == (
        DemError(probability=0.1, detectors=(0, 2), observables=(0, 1)),
        DemError(probability=0.2, detectors=(1,), observables=()),
    )
    assert model.detector_error_matrix().tolist() == [[1, 0], [0, 1], [1, 0]]
    assert model.observables_flips_matrix().tolist() == [[1, 0], [1, 0]]


def test_shape_comes_from_the_declarations_not_the_largest_index() -> None:
    """The last detectors are declared but no error mentions them."""

    text = "error(0.1) D0\ndetector D0\ndetector D1\ndetector D2\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3
    assert model.num_observables == 0
    assert model.errors == (DemError(probability=0.1, detectors=(0,), observables=()),)


def test_parses_coordinate_declarations_by_ignoring_the_coordinates() -> None:
    """The declaration's index is the target, never the coordinate."""

    text = "error(0.1) D0\ndetector(7, 8) D0\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 1
    assert model.errors == (DemError(probability=0.1, detectors=(0,), observables=()),)


def test_rejects_an_unrecognized_instruction() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        DetectorErrorModel.from_stim_text("noise 0.1\n")


def test_rejects_coordinates_on_a_logical_observable() -> None:
    """Only a detector declaration carries coordinates."""

    with pytest.raises(ValueError, match="exactly one L index"):
        DetectorErrorModel.from_stim_text("logical_observable(7, 8) L0\n")


def test_rejects_declarations_that_skip_an_index() -> None:
    """A hole in the declarations would silently misstate the shape."""

    with pytest.raises(ValueError, match="consecutive"):
        DetectorErrorModel.from_stim_text("detector D1\n")


def test_rejects_an_error_outside_the_declared_shape() -> None:
    with pytest.raises(ValueError, match="outside the model shape"):
        DetectorErrorModel.from_stim_text("error(0.1) D5\ndetector D0\n")


def test_from_text_orders_errors_that_tie_on_detectors_and_probability() -> None:
    """The canonical key separates errors by their observables.

    Two errors can share a probability and a detector signature, which makes
    the observable component of the sort key the only thing that orders them.
    ``sorted`` is stable, so a key that drops that component leaves the two in
    whatever order the text listed them.
    """

    text = (
        "error(0.25) D1 L1\n"
        "error(0.25) D1 L0\n"
        "error(0.5) D0\n"
        "detector D0\n"
        "detector D1\n"
        "logical_observable L0\n"
        "logical_observable L1\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.errors == (
        DemError(probability=0.5, detectors=(0,), observables=()),
        DemError(probability=0.25, detectors=(1,), observables=(0,)),
        DemError(probability=0.25, detectors=(1,), observables=(1,)),
    )
