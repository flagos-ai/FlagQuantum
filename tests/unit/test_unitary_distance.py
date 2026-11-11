"""Phase-insensitive distance is distinct from amplitude conformance."""

import warnings
from typing import Any

import numpy as np
import pytest

from flagquantum.simulation.numerics.unitary_distance import (
    unitary_distance_ignoring_global_phase,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("phase", [0.0, 0.31, np.pi])
def test_metric_matches_original_for_phase_and_relative_changes(
    dimension: int, phase: float
) -> None:
    from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase as source

    rng = np.random.RandomState(42)
    left = np.linalg.qr(
        rng.randn(dimension, dimension) + 1j * rng.randn(dimension, dimension)
    )[0]
    right = np.exp(1j * phase) * left
    before = left.copy(), right.copy()
    assert unitary_distance_ignoring_global_phase(left, right) == source(left, right)
    assert unitary_distance_ignoring_global_phase(left, right) < 3e-8
    right[:, 0] *= -1
    assert unitary_distance_ignoring_global_phase(left, right) == source(left, right)
    if dimension > 1:
        assert unitary_distance_ignoring_global_phase(left, right) > 0.5
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(-right[:, 0], before[1][:, 0])


@pytest.mark.parametrize("shapes", [((2, 2), (3, 3)), ((2, 3), (2, 3)), ((2,), (2,))])
def test_shape_errors_match_original(
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase as source

    left, right = (np.zeros(shape) for shape in shapes)
    with pytest.raises(ValueError) as original:
        source(left, right)
    with pytest.raises(ValueError) as native:
        unitary_distance_ignoring_global_phase(left, right)
    assert str(native.value) == str(original.value)


def test_clamp_and_nonunitary_boundary_are_not_extra_validation() -> None:
    from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase as source

    identity = np.eye(2)
    for right in (identity * (1 + 1e-14), identity * 2, identity * 0):
        assert unitary_distance_ignoring_global_phase(identity, right) == source(
            identity, right
        )
    assert unitary_distance_ignoring_global_phase(identity, identity * 2) == 0


def test_global_phase_cannot_replace_exact_or_controlled_conformance() -> None:
    identity = np.eye(2, dtype=np.complex128)
    phaseful = np.exp(0.3j) * identity
    assert unitary_distance_ignoring_global_phase(identity, phaseful) < 3e-8
    assert not np.allclose(identity, phaseful)
    controlled_identity = np.eye(4, dtype=np.complex128)
    controlled_phase = controlled_identity.copy()
    controlled_phase[2:, 2:] = phaseful
    assert (
        unitary_distance_ignoring_global_phase(controlled_identity, controlled_phase)
        > 0.1
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.complex64,
        np.complex128,
        np.clongdouble,
        object,
    ],
)
def test_scalar_type_and_precision_match_source(dtype: Any) -> None:
    from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase as source

    left = np.eye(3, dtype=dtype)
    right = np.diag(np.array([1, 0, 0], dtype=dtype))
    expected = source(left, right)
    actual = unitary_distance_ignoring_global_phase(left, right)
    assert type(actual) is type(expected)
    assert actual == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_clamp_and_warning_sequence_match_source(value: float) -> None:
    from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase as source

    outcomes = []
    for evaluator in (source, unitary_distance_ignoring_global_phase):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = evaluator(np.eye(2), np.eye(2) * value)
        outcomes.append((result, [(w.category, str(w.message)) for w in seen]))
    assert outcomes[0] == outcomes[1]


def test_empty_square_retains_division_warning_and_scalar_type() -> None:
    from qualtran.linalg.matrix import unitary_distance_ignoring_global_phase as source

    empty = np.empty((0, 0))
    with pytest.warns(RuntimeWarning, match="invalid value"):
        expected = source(empty, empty)
    with pytest.warns(RuntimeWarning, match="invalid value"):
        actual = unitary_distance_ignoring_global_phase(empty, empty)
    assert type(actual) is type(expected) and actual == expected == 0


@pytest.mark.parametrize("dimension", [2, 3])
def test_original_matrix_test_with_native_replacement(dimension: int) -> None:
    from cirq.testing import random_unitary

    # Original matrix_test.py scenario, with only the metric replaced.
    rng = np.random.RandomState(1234)
    for _ in range(3):
        left = random_unitary(dimension, random_state=rng)
        right = np.exp(1j * rng.random()) * left
        np.testing.assert_almost_equal(
            unitary_distance_ignoring_global_phase(left, right), 0
        )
