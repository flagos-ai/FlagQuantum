"""FFT-grid source parity does not imply a full unit-circle certificate."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy import fft as scipy_fft

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.simulation.numerics.qsp_grid import (
    assert_is_qsp_polynomial,
    polynomial_max_abs_value_on_unit_circle,
    scale_down_to_qsp_polynomial,
)
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "coefficients",
    [(), (0.0,), (0.25, 0.25), (1.0, -2.0), (1, -2), (0.3j, 0.75, -0.25j)],
)
@pytest.mark.parametrize("n_points", [4, 16])
def test_grid_scaling_and_assertion_match_source(
    coefficients: tuple[complex, ...], n_points: int
) -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        _polynomial_max_abs_value_on_unit_circle as source_max,
    )
    from qualtran.linalg.polynomial.qsp_testing import (
        assert_is_qsp_polynomial as source_assert,
    )
    from qualtran.linalg.polynomial.qsp_testing import (
        scale_down_to_qsp_polynomial as source_scale,
    )

    actual = polynomial_max_abs_value_on_unit_circle(
        coefficients, fft=scipy_fft.fft, n_points=n_points
    )
    assert actual == source_max(coefficients, n_points=n_points)
    scaled = scale_down_to_qsp_polynomial(
        coefficients, fft=scipy_fft.fft, n_points=n_points
    )
    np.testing.assert_array_equal(scaled, source_scale(coefficients, n_points=n_points))
    assert scaled.dtype == source_scale(coefficients, n_points=n_points).dtype
    if actual <= 1:
        source_assert(coefficients, n_points=n_points)
        assert_is_qsp_polynomial(coefficients, fft=scipy_fft.fft, n_points=n_points)
    else:
        with pytest.raises(AssertionError) as original:
            source_assert(coefficients, n_points=n_points)
        with pytest.raises(AssertionError) as native:
            assert_is_qsp_polynomial(coefficients, fft=scipy_fft.fft, n_points=n_points)
        assert str(native.value) == str(original.value)


def test_fft_call_padding_dtype_and_consumer_replacement() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        scale_down_to_qsp_polynomial as source,
    )

    calls: list[NDArray[np.float64 | np.complex128 | np.int64]] = []

    def record(
        data: NDArray[np.float64 | np.complex128 | np.int64],
    ) -> NDArray[np.complex128]:
        calls.append(data.copy())
        result: NDArray[np.complex128] = scipy_fft.fft(data)
        return result

    coefficients = np.array([1.0, -2.0])
    expected = source(coefficients, n_points=8)
    actual = scale_down_to_qsp_polynomial(coefficients, fft=record, n_points=8)
    assert len(calls) == 1 and calls[0].dtype == coefficients.dtype
    np.testing.assert_array_equal(
        calls[0], np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    transformed: NDArray[np.complex128] = scipy_fft.fft(calls[0])

    def cached(
        data: NDArray[np.float64 | np.complex128 | np.int64],
    ) -> NDArray[np.complex128]:
        np.testing.assert_array_equal(data, calls[0])
        return transformed.copy()

    for provider in (record, cached):
        scaled = scale_down_to_qsp_polynomial(coefficients, fft=provider, n_points=8)
        np.testing.assert_array_equal(scaled, expected)
        unitary = np.diag(np.exp(1j * np.array([0.31, -0.73])))
        matrix = evaluate_polynomial_of_unitary_matrix(
            tuple(complex(c) for c in scaled), unitary
        )
        expected_matrix = expected[0] * np.eye(2) + expected[1] * unitary
        np.testing.assert_array_equal(matrix, expected_matrix)
        recipe = Circuit(2, dtype=torch.complex128).h(0).rz(0, theta=0.2).to_ir()
        circuit = Circuit(2, dtype=torch.complex128).append(
            FieldOperation(2, recipe.instructions)
        )
        assert circuit.to_ir().instructions == recipe.instructions
        state = circuit.state().detach().numpy().reshape(2, 2)
        np.testing.assert_array_equal(matrix @ state, expected_matrix @ state)
        np.testing.assert_array_equal((matrix @ state)[:, 1], 0)
        assert len(recipe.instructions) == 2
    np.testing.assert_array_equal(coefficients, [1.0, -2.0])
    assert not np.shares_memory(actual, coefficients)
    unchanged = np.array([0.25, 0.25])
    assert (
        scale_down_to_qsp_polynomial(unchanged, fft=scipy_fft.fft, n_points=8)
        is unchanged
    )


def test_finite_grid_misses_between_grid_peak_in_both_implementations() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        assert_is_qsp_polynomial as source_assert,
    )
    from qualtran.linalg.polynomial.qsp_testing import (
        scale_down_to_qsp_polynomial as source_scale,
    )

    coefficients = np.array([0.53, 0.53 * np.exp(1j * np.pi / 4)])
    source_assert(coefficients, n_points=4)
    assert_is_qsp_polynomial(coefficients, fft=scipy_fft.fft, n_points=4)
    np.testing.assert_array_equal(
        scale_down_to_qsp_polynomial(coefficients, fft=scipy_fft.fft, n_points=4),
        source_scale(coefficients, n_points=4),
    )
    point = np.exp(-1j * np.pi / 4)
    assert abs(coefficients[0] + coefficients[1] * point) > 1.05
    assert (
        polynomial_max_abs_value_on_unit_circle(
            coefficients, fft=scipy_fft.fft, n_points=4
        )
        < 1
    )


@pytest.mark.parametrize("n_points", [0, 1, -1])
def test_invalid_grid_errors_match_source(n_points: int) -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        _polynomial_max_abs_value_on_unit_circle as source,
    )

    try:
        source((1.0, 2.0), n_points=n_points)
    except (ValueError, IndexError) as error:
        with pytest.raises(type(error)) as native:
            polynomial_max_abs_value_on_unit_circle(
                (1.0, 2.0), fft=scipy_fft.fft, n_points=n_points
            )
        assert str(native.value) == str(error)
    else:
        pytest.fail("source unexpectedly accepted invalid grid")


def test_nan_and_provider_failure_are_not_repaired() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        scale_down_to_qsp_polynomial as source,
    )

    data = np.array([np.nan])
    np.testing.assert_array_equal(
        scale_down_to_qsp_polynomial(data, fft=scipy_fft.fft, n_points=4),
        source(data, n_points=4),
    )
    with pytest.raises(AssertionError, match="nan"):
        assert_is_qsp_polynomial(data, fft=scipy_fft.fft, n_points=4)

    def refuse(
        data: NDArray[np.float64 | np.complex128 | np.int64],
    ) -> NDArray[np.complex128]:
        raise ArithmeticError("test FFT failure")

    with pytest.raises(ArithmeticError, match="test FFT failure"):
        scale_down_to_qsp_polynomial((1.0,), fft=refuse, n_points=4)
