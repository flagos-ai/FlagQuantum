"""Default Bessel call-chain parity, optional dependencies and original consumers."""

import builtins
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import sympy
from numpy.typing import NDArray
from scipy import special

from flagquantum.simulation.numerics.jacobi_anger_coefficients import (
    approx_exp_cos_by_jacobi_anger,
    approx_exp_sin_by_jacobi_anger,
)
from flagquantum.simulation.numerics.jacobi_anger_degree import (
    degree_jacobi_anger_approximation,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("sine", [False, True])
@pytest.mark.parametrize("degree", [-2, 0, 5])
@pytest.mark.parametrize("t", [np.float32(2), np.float64(-3), 2 + 0.5j, np.nan, np.inf])
def test_default_coefficients_dtype_values_warnings_and_errors_match_source(
    sine: bool, degree: int, t: Any
) -> None:
    from qualtran.linalg.polynomial import jacobi_anger_approximations as source

    original = (
        source.approx_exp_sin_by_jacobi_anger
        if sine
        else source.approx_exp_cos_by_jacobi_anger
    )
    native = approx_exp_sin_by_jacobi_anger if sine else approx_exp_cos_by_jacobi_anger
    outcomes = []
    for evaluator in (original, native):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = evaluator(t, degree=degree)
        outcomes.append((result, [(w.category, str(w.message)) for w in seen]))
    expected, actual = outcomes
    assert actual[1] == expected[1] and actual[0].dtype == expected[0].dtype
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize(
    "t, precision", [(0.0, 1e-5), (-3.0, 1e-7), (2.0, 1e-10), (1.0, 1.0)]
)
def test_default_degree_records_exact_bessel_visits(
    monkeypatch: pytest.MonkeyPatch, t: float, precision: float
) -> None:
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        degree_jacobi_anger_approximation as source,
    )

    jv = special.jv
    calls: list[tuple[int, float]] = []

    def record(order: int, argument: float) -> float:
        calls.append((order, argument))
        result: float = jv(order, argument)
        return result

    monkeypatch.setattr(special, "jv", record)
    visits = []
    outcomes = []
    for evaluator in (source, degree_jacobi_anger_approximation):
        calls.clear()
        value: object
        try:
            value = evaluator(t, precision=precision)
        except AssertionError as error:
            value = type(error), str(error)
        outcomes.append(value)
        visits.append(calls.copy())
    assert outcomes[0] == outcomes[1] and visits[0] == visits[1]


@pytest.mark.parametrize("operation", ["cos", "sin", "degree"])
def test_optional_bessel_import_failure_and_provider_bypass(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    saved_import: Callable[..., ModuleType] = builtins.__import__
    imports: list[str] = []

    def gated_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "scipy.special":
            imports.append(name)
            raise ModuleNotFoundError("test absent SciPy Bessel")
        return saved_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", gated_import)
    if operation == "degree":
        with pytest.raises(ModuleNotFoundError, match="test absent SciPy Bessel"):
            degree_jacobi_anger_approximation(2, precision=1e-5)
        assert len(imports) == 1
        degree_jacobi_anger_approximation(2, precision=1e-5, bessel_jn=special.jv)
    else:
        factory = (
            approx_exp_cos_by_jacobi_anger
            if operation == "cos"
            else approx_exp_sin_by_jacobi_anger
        )
        with pytest.raises(ModuleNotFoundError, match="test absent SciPy Bessel"):
            factory(2, degree=4)
        assert len(imports) == 1
        factory(2, degree=4, bessel_jn=special.jv)
    assert len(imports) == 1
    variable = sympy.Symbol("t", real=True)
    assert isinstance(
        degree_jacobi_anger_approximation(variable, precision=1e-5), sympy.Expr
    )
    assert len(imports) == 1


@pytest.mark.parametrize("sine", [False, True])
def test_default_vector_provider_called_once_after_orders_created(
    monkeypatch: pytest.MonkeyPatch, sine: bool
) -> None:
    calls: list[tuple[NDArray[np.int64], float]] = []
    jv = special.jv

    def record(orders: NDArray[np.int64], t: float) -> NDArray[np.float64]:
        calls.append((orders.copy(), t))
        result: NDArray[np.float64] = jv(orders, t)
        return result

    monkeypatch.setattr(special, "jv", record)
    factory = approx_exp_sin_by_jacobi_anger if sine else approx_exp_cos_by_jacobi_anger
    factory(2, degree=4)
    assert len(calls) == 1 and calls[0][1] == 2
    np.testing.assert_array_equal(calls[0][0], np.arange(-4, 5))


@pytest.mark.parametrize("sine", [False, True])
@pytest.mark.parametrize("t", [2.0, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("precision", [1e-5, 1e-7, 1e-10])
def test_original_jacobi_anger_tests_with_native_default_bindings(
    monkeypatch: pytest.MonkeyPatch, sine: bool, t: float, precision: float
) -> None:
    from qualtran.linalg.polynomial import (
        jacobi_anger_approximations_test as source_test,
    )

    monkeypatch.setattr(
        source_test,
        "degree_jacobi_anger_approximation",
        degree_jacobi_anger_approximation,
    )
    monkeypatch.setattr(
        source_test, "approx_exp_cos_by_jacobi_anger", approx_exp_cos_by_jacobi_anger
    )
    monkeypatch.setattr(
        source_test, "approx_exp_sin_by_jacobi_anger", approx_exp_sin_by_jacobi_anger
    )
    consumer: Callable[[float, float], None] = (
        source_test.test_exp_sin_approximation
        if sine
        else source_test.test_exp_cos_approximation
    )
    consumer(t, precision)
