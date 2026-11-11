"""Source Laurent coefficients connect to the existing phaseful evaluator."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.special import jv

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.simulation.numerics.jacobi_anger_coefficients import (
    approx_exp_cos_by_jacobi_anger,
    approx_exp_sin_by_jacobi_anger,
)
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("t", [-3.0, 0.0, 2.0, 10.0])
@pytest.mark.parametrize("degree", [-1, 0, 1, 8])
def test_coefficients_dtype_and_provider_call_match_source(
    t: float, degree: int
) -> None:
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        approx_exp_cos_by_jacobi_anger as source_cos,
    )
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        approx_exp_sin_by_jacobi_anger as source_sin,
    )

    calls: list[tuple[NDArray[np.int64], float]] = []

    def provider(orders: NDArray[np.int64], argument: float) -> NDArray[np.float64]:
        calls.append((orders.copy(), argument))
        result: NDArray[np.float64] = jv(orders, argument)
        return result

    cosine = approx_exp_cos_by_jacobi_anger(t, degree=degree, bessel_jn=provider)
    sine = approx_exp_sin_by_jacobi_anger(t, degree=degree, bessel_jn=provider)
    np.testing.assert_array_equal(cosine, source_cos(t, degree=degree))
    np.testing.assert_array_equal(sine, source_sin(t, degree=degree))
    assert cosine.dtype == np.complex128 and sine.dtype == np.float64
    assert len(calls) == 2
    for orders, argument in calls:
        np.testing.assert_array_equal(orders, np.arange(-degree, degree + 1))
        assert argument == t


@pytest.mark.parametrize("sine", [False, True])
def test_callback_replacement_and_full_complex_evaluation(sine: bool) -> None:
    from qualtran.linalg.polynomial.basic import (
        evaluate_polynomial_of_unitary_matrix as source_evaluate,
    )
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        approx_exp_cos_by_jacobi_anger as source_cos,
    )
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        approx_exp_sin_by_jacobi_anger as source_sin,
    )

    t, degree = 2.0, 14
    orders = np.arange(-degree, degree + 1)
    table: NDArray[np.float64] = jv(orders, t)

    def cached(indices: NDArray[np.int64], argument: float) -> NDArray[np.float64]:
        assert argument == t
        np.testing.assert_array_equal(indices, orders)
        return table.copy()

    factory = approx_exp_sin_by_jacobi_anger if sine else approx_exp_cos_by_jacobi_anger
    source = source_sin if sine else source_cos
    original = source(t, degree=degree)
    # Replace only the Bessel evaluator; coefficient and matrix consumers stay identical.
    for provider in (jv, cached):
        coefficients = factory(t, degree=degree, bessel_jn=provider)
        np.testing.assert_array_equal(coefficients, original)
        angles = np.array([-0.71, 0.23])
        unitary = np.diag(np.exp(1j * angles))
        result = evaluate_polynomial_of_unitary_matrix(
            tuple(coefficients), unitary, offset=-degree
        )
        np.testing.assert_array_equal(
            result, source_evaluate(original, unitary, offset=-degree)
        )
        expected = np.diag(
            np.exp(1j * t * (np.sin(angles) if sine else np.cos(angles)))
        )
        np.testing.assert_allclose(result, expected, atol=2e-12, rtol=0)
        recipe = Circuit(2, dtype=torch.complex128).h(0).rz(0, theta=0.31).to_ir()
        circuit = Circuit(2, dtype=torch.complex128).append(
            FieldOperation(2, recipe.instructions)
        )
        assert circuit.to_ir().instructions == recipe.instructions
        state = circuit.state().detach().numpy().reshape(2, 2)
        evolved = result @ state
        np.testing.assert_allclose(evolved, expected @ state, atol=2e-12, rtol=0)
        np.testing.assert_array_equal(evolved[:, 1], 0)
        assert len(recipe.instructions) == 2


def test_provider_failure_is_not_swallowed() -> None:
    def refuse(orders: NDArray[np.int64], t: float) -> NDArray[np.float64]:
        raise ArithmeticError("test Bessel failure")

    for factory in (approx_exp_sin_by_jacobi_anger, approx_exp_cos_by_jacobi_anger):
        with pytest.raises(ArithmeticError, match="test Bessel failure"):
            factory(2.0, degree=3, bessel_jn=refuse)
