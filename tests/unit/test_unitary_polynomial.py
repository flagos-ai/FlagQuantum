"""Unmodified-source parity for dense Laurent polynomial inspection."""

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
