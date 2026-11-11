"""Source random draw consumption and finite GQSP pair checks."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from flagquantum.simulation.numerics.qsp_sampling import (
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("degree", [0, 1, 2, 7])
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("seed", [0, 17])
def test_random_coefficients_and_rng_match_source(
    degree: int, real: bool, seed: int
) -> None:
    from qualtran.linalg.polynomial.qsp_testing import random_qsp_polynomial as source

    native_rng, original_rng = np.random.RandomState(seed), np.random.RandomState(seed)
    actual = random_qsp_polynomial(
        degree, random_state=native_rng, only_real_coeffs=real
    )
    expected = source(degree, random_state=original_rng, only_real_coeffs=real)
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == degree
    assert sum(abs(value) for value in actual) <= 1
    np.testing.assert_array_equal(native_rng.random(9), original_rng.random(9))


def test_negative_degree_refuses_before_random_consumption() -> None:
    from qualtran.linalg.polynomial.qsp_testing import random_qsp_polynomial as source

    native_rng, original_rng = np.random.RandomState(9), np.random.RandomState(9)
    with pytest.raises(ValueError):
        source(-1, random_state=original_rng)
    with pytest.raises(ValueError):
        random_qsp_polynomial(-1, random_state=native_rng)
    np.testing.assert_array_equal(native_rng.random(9), original_rng.random(9))


@pytest.mark.parametrize("n_points", [0, 1, 32])
@pytest.mark.parametrize("valid", [False, True])
def test_pair_check_pass_failure_and_rng_match_source(
    n_points: int, valid: bool
) -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle as source,
    )

    left, right = ((0.5, 0.5), (0.5, -0.5)) if valid else ((2.0,), (0.0,))
    native_rng, original_rng = np.random.RandomState(13), np.random.RandomState(13)
    if valid or n_points == 0:
        source(left, right, random_state=original_rng, n_points=n_points)
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
            left, right, random_state=native_rng, n_points=n_points
        )
    else:
        with pytest.raises(AssertionError) as original:
            source(left, right, random_state=original_rng, n_points=n_points)
        with pytest.raises(AssertionError) as native:
            check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
                left, right, random_state=native_rng, n_points=n_points
            )
        assert str(native.value) == str(original.value)
    np.testing.assert_array_equal(native_rng.random(9), original_rng.random(9))


def test_invalid_construction_fails_before_sampling() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle as source,
    )

    native_rng, original_rng, expected_rng = (
        np.random.RandomState(19) for _ in range(3)
    )
    for check, rng in (
        (source, original_rng),
        (check_gqsp_polynomial_pair_on_random_points_on_unit_circle, native_rng),
    ):
        with pytest.raises(ValueError):
            check([], (0.0,), random_state=rng, n_points=8)
    expected = expected_rng.random(9)
    np.testing.assert_array_equal(native_rng.random(9), expected)
    np.testing.assert_array_equal(original_rng.random(9), expected)


def test_source_polynomial_object_counterexample() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle as source,
    )

    original_rng, expected_rng = np.random.RandomState(19), np.random.RandomState(19)
    # The source annotation accepts Polynomial, but wrapping it again fails after sampling.
    with pytest.raises(TypeError, match="Polynomial"):
        source(Polynomial([1.0]), (0.0,), random_state=original_rng, n_points=8)
    expected_rng.random(8)
    np.testing.assert_array_equal(original_rng.random(9), expected_rng.random(9))


def test_custom_tolerance_is_not_replaced_by_normalization() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle as source,
    )

    for check in (source, check_gqsp_polynomial_pair_on_random_points_on_unit_circle):
        check(
            (1.001,),
            (0.0,),
            random_state=np.random.RandomState(4),
            rtol=0.01,
            n_points=16,
        )
        with pytest.raises(AssertionError):
            check(
                (1.001,),
                (0.0,),
                random_state=np.random.RandomState(4),
                rtol=1e-7,
                n_points=16,
            )
