"""Unmodified-source parity for dense Laurent polynomial inspection."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dimension", [1, 2, 4])
@pytest.mark.parametrize("offset", [-3, 0, 2])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_polynomial_matches_source(
    dimension: int, offset: int, dtype: type[np.complex64] | type[np.complex128]
) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    rng = np.random.RandomState(17)
    raw = np.linalg.qr(
        rng.randn(dimension, dimension) + 1j * rng.randn(dimension, dimension)
    )[0]
    unitary = (
        raw.astype(np.complex64) if dtype is np.complex64 else raw.astype(np.complex128)
    )
    before = unitary.copy()
    coefficients = (0.25 + 0.5j, -0.75j, 0.0, 1.25)
    actual = evaluate_polynomial_of_unitary_matrix(coefficients, unitary, offset=offset)
    np.testing.assert_array_equal(actual, source(coefficients, unitary, offset=offset))
    assert actual.dtype == unitary.dtype
    # Independent spectral evaluation also checks the negative-power convention.
    eigenvalues, vectors = np.linalg.eig(unitary)
    values = sum(c * eigenvalues ** (i + offset) for i, c in enumerate(coefficients))
    expected = vectors @ np.diag(values) @ np.linalg.inv(vectors)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(unitary, before)


@pytest.mark.parametrize("offset", [-2, 0, 3])
def test_empty_polynomial_and_real_dtype(offset: int) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    unitary = np.array([[0.0, -1.0], [1.0, 0.0]])
    for coefficients in ((), (1.0, 2.0, -3.0)):
        actual = evaluate_polynomial_of_unitary_matrix(
            coefficients, unitary, offset=offset
        )
        np.testing.assert_array_equal(
            actual, source(coefficients, unitary, offset=offset)
        )
        assert actual.dtype == unitary.dtype


@pytest.mark.parametrize("shape", [(2,), (2, 3)])
def test_invalid_shape_preserves_source_assertion(shape: tuple[int, ...]) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    for evaluator in (source, evaluate_polynomial_of_unitary_matrix):
        with pytest.raises(AssertionError):
            evaluator((1.0,), np.zeros(shape))


def test_negative_offset_uses_adjoint_even_for_nonunitary_input() -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    matrix = np.array([[2.0, 1.0j], [0.0, 3.0]])
    actual = evaluate_polynomial_of_unitary_matrix((1.0,), matrix, offset=-1)
    np.testing.assert_array_equal(actual, source((1.0,), matrix, offset=-1))
    np.testing.assert_array_equal(actual, matrix.conj().T)
    assert not np.allclose(actual, np.linalg.inv(matrix))


def test_complex_coefficients_do_not_silently_promote_real_storage() -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    for evaluator in (source, evaluate_polynomial_of_unitary_matrix):
        with pytest.raises(TypeError):
            evaluator((1j,), np.eye(2))


def test_append_workflow_remains_phaseful_and_clean() -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    recipe = Circuit(2, dtype=torch.complex128).h(0).rz(0, theta=0.37).to_ir()
    operation = FieldOperation(2, recipe.instructions)
    columns: list[NDArray[np.complex128]] = []
    for bit in (0, 1):
        circuit = Circuit(2, dtype=torch.complex128)
        if bit:
            circuit.x(0)
        circuit.append(operation)
        amplitudes = circuit.state().detach().numpy().reshape(2, 2)
        np.testing.assert_array_equal(amplitudes[:, 1], 0)
        columns.append(amplitudes[:, 0])
        assert (
            circuit.to_ir().instructions[-len(recipe.instructions) :]
            == recipe.instructions
        )
    unitary = np.column_stack(columns)
    result = evaluate_polynomial_of_unitary_matrix(
        (0.5, 0.25j, -0.5), unitary, offset=-1
    )
    np.testing.assert_array_equal(
        result, source((0.5, 0.25j, -0.5), unitary, offset=-1)
    )
    # Numerical postprocessing leaves the two emitted gates and idle clean wire alone.
    assert len(recipe.instructions) == 2


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int64,
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
@pytest.mark.parametrize("offset", [-2, 2])
def test_numpy_dtype_results_failures_and_warnings_match_source(
    dtype: Any, offset: int
) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    matrix = np.array([[0, 1], [1, 0]], dtype=dtype)
    before = matrix.copy()
    outcomes: list[tuple[type[Exception] | type[None], str, NDArray[Any] | None]] = []
    for evaluator in (source, evaluate_polynomial_of_unitary_matrix):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                result = evaluator((1, 2, -3), matrix, offset=offset)
            except (TypeError, ValueError, OverflowError) as error:
                outcomes.append((type(error), str(error), None))
            else:
                outcomes.append((type(None), "", result))
        assert not seen
        np.testing.assert_array_equal(matrix, before)
    expected, actual = outcomes
    assert actual[:2] == expected[:2]
    if expected[2] is not None:
        assert actual[2] is not None
        np.testing.assert_array_equal(actual[2], expected[2])
        assert actual[2].dtype == expected[2].dtype
        assert type(actual[2]) is type(expected[2])


@pytest.mark.parametrize("kind", ["subclass", "masked", "matrix"])
def test_numpy_subclass_dispatch_and_return_type_match_source(kind: str) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    class Matrix(np.ndarray[tuple[int, ...], np.dtype[np.complex128]]):
        pass

    raw = np.array([[1j, 0], [0, -1j]], dtype=np.complex128)
    # NumPy masked-array construction is an untyped external boundary.
    masked: Callable[[NDArray[np.complex128]], NDArray[Any]] = np.ma.array
    factories: dict[str, Callable[[], NDArray[Any]]] = {
        "subclass": lambda: raw.view(Matrix),
        "masked": lambda: masked(raw),
        "matrix": lambda: np.matrix(raw),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        matrix = factories[kind]()
    outcomes = []
    for evaluator in (source, evaluate_polynomial_of_unitary_matrix):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = evaluator((1, 2j, 3), matrix, offset=-1)
        outcomes.append((result, [(w.category, str(w.message)) for w in seen]))
    expected, actual = outcomes
    assert actual[1] == expected[1] and type(actual[0]) is type(expected[0])
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("offset", [-1, 0, 1.5])
def test_empty_matrix_and_invalid_power_match_source(offset: Any) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source,
    )

    empty = np.empty((0, 0), dtype=np.complex128)
    if isinstance(offset, int):
        np.testing.assert_array_equal(
            evaluate_polynomial_of_unitary_matrix((1, 2), empty, offset=offset),
            source((1, 2), empty, offset=offset),
        )
    else:
        with pytest.raises(TypeError) as expected:
            source((), empty, offset=offset)
        with pytest.raises(TypeError) as actual:
            evaluate_polynomial_of_unitary_matrix((), empty, offset=offset)
        assert str(actual.value) == str(expected.value)


def test_original_basic_test_with_native_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.linalg.polynomial import basic_test

    monkeypatch.setattr(
        basic_test,
        "evaluate_polynomial_of_unitary_matrix",
        evaluate_polynomial_of_unitary_matrix,
    )
    basic_test.test_evaluate_polynomial_of_unitary_matrix()
