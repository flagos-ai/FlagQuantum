"""Whole-source QSP inspection parity, including implicit FFT and failures."""

import builtins
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.fft import fft

from flagquantum.simulation.numerics.qsp_grid import (
    assert_is_qsp_polynomial,
    polynomial_max_abs_value_on_unit_circle,
    scale_down_to_qsp_polynomial,
)
from flagquantum.simulation.numerics.qsp_sampling import (
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
)
from flagquantum.simulation.value_dimensions import SymbolicArrayShape

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


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
@pytest.mark.parametrize("operation", ["maximum", "scale", "assert"])
def test_default_fft_dtype_scalar_and_failure_match_source(
    dtype: Any, operation: str
) -> None:
    from qualtran.linalg.polynomial import qsp_testing as source

    originals: dict[str, Callable[..., object]] = {
        "maximum": source._polynomial_max_abs_value_on_unit_circle,
        "scale": source.scale_down_to_qsp_polynomial,
        "assert": source.assert_is_qsp_polynomial,
    }
    natives: dict[str, Callable[..., object]] = {
        "maximum": polynomial_max_abs_value_on_unit_circle,
        "scale": scale_down_to_qsp_polynomial,
        "assert": assert_is_qsp_polynomial,
    }
    data = np.array([0.75, 0.5], dtype=dtype)
    outcomes = []
    for evaluator in (originals[operation], natives[operation]):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            value: object
            try:
                value = evaluator(data, n_points=8)
            except (ValueError, TypeError, AssertionError) as error:
                value = (type(error), str(error))
        outcomes.append((value, [(w.category, str(w.message)) for w in seen]))
    (expected, source_warnings), (actual, native_warnings) = outcomes
    assert type(actual) is type(expected) and native_warnings == source_warnings
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", ["polynomial", "shape", "empty", "matrix", "scalar"])
@pytest.mark.parametrize("n_points", [0, 8])
def test_pair_foreign_inputs_keep_failure_and_rng_order(
    kind: str, n_points: int
) -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle as source,
    )
    from qualtran.symbolics import Shaped

    factories: dict[str, Callable[[], Any]] = {
        "polynomial": lambda: Polynomial([1.0]),
        "shape": lambda: Shaped((2,)),
        "empty": list,
        "matrix": lambda: np.ones((2, 2)),
        "scalar": lambda: np.array(1.0),
    }
    outcome: tuple[type[Exception] | type[None], str]
    outcomes = []
    draws = []
    left = factories[kind]()
    for evaluator in (
        source,
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    ):
        rng = np.random.RandomState(71)
        try:
            evaluator(left, [0.0], random_state=rng, n_points=n_points)
        except (TypeError, ValueError, AssertionError) as error:
            outcome = type(error), str(error)
        else:
            outcome = type(None), ""
        outcomes.append(outcome)
        draws.append(rng.random(9))
    assert outcomes[0] == outcomes[1]
    np.testing.assert_array_equal(draws[0], draws[1])


def test_native_shape_descriptor_preserves_source_failure_stage() -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle as source,
    )
    from qualtran.symbolics import Shaped

    outcomes = []
    for evaluator, value in (
        (source, Shaped((2,))),
        (
            check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
            SymbolicArrayShape((2,)),
        ),
    ):
        rng = np.random.RandomState(71)
        with pytest.raises(TypeError) as error:
            evaluator(value, [0.0], random_state=rng, n_points=8)
        outcomes.append(str(error.value).replace("SymbolicArrayShape", "Shaped"))
        expected_rng = np.random.RandomState(71)
        expected_rng.random(8)
        np.testing.assert_array_equal(rng.random(9), expected_rng.random(9))
    # Only the native class name differs; no data or phase is fabricated.
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("operation", ["maximum", "scale", "assert"])
def test_missing_optional_fft_and_provider_bypass_preserve_order(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from qualtran.linalg.polynomial import qsp_testing as source

    originals: dict[str, Callable[..., object]] = {
        "maximum": source._polynomial_max_abs_value_on_unit_circle,
        "scale": source.scale_down_to_qsp_polynomial,
        "assert": source.assert_is_qsp_polynomial,
    }
    natives: dict[str, Callable[..., object]] = {
        "maximum": polynomial_max_abs_value_on_unit_circle,
        "scale": scale_down_to_qsp_polynomial,
        "assert": assert_is_qsp_polynomial,
    }
    events: list[str] = []

    class ArrayInput:
        def __array__(
            self, dtype: np.dtype[Any] | None = None, copy: bool | None = None
        ) -> NDArray[np.float64]:
            events.append("array")
            return np.array([0.25])

    saved_import: Callable[..., ModuleType] = builtins.__import__

    def gated_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "scipy.fft":
            events.append("import")
            raise ModuleNotFoundError("test missing SciPy FFT")
        return saved_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", gated_import)
    for evaluator in (originals[operation], natives[operation]):
        events.clear()
        with pytest.raises(ModuleNotFoundError, match="test missing SciPy FFT"):
            evaluator(ArrayInput(), n_points=8)
        assert events == (["array", "import"] if operation == "scale" else ["import"])

    def provider(data: NDArray[Any]) -> NDArray[Any]:
        events.append("fft")
        result: NDArray[Any] = fft(data)
        return result

    events.clear()
    natives[operation](ArrayInput(), fft=provider, n_points=8)
    assert events == ["array", "fft"]


@pytest.mark.parametrize("operation", ["maximum", "scale", "assert"])
@pytest.mark.parametrize("value", [Polynomial([1.0]), np.array(1.0), np.ones((2, 2))])
def test_grid_foreign_inputs_match_source(operation: str, value: Any) -> None:
    from qualtran.linalg.polynomial import qsp_testing as source

    originals: dict[str, Callable[..., object]] = {
        "maximum": source._polynomial_max_abs_value_on_unit_circle,
        "scale": source.scale_down_to_qsp_polynomial,
        "assert": source.assert_is_qsp_polynomial,
    }
    natives: dict[str, Callable[..., object]] = {
        "maximum": polynomial_max_abs_value_on_unit_circle,
        "scale": scale_down_to_qsp_polynomial,
        "assert": assert_is_qsp_polynomial,
    }
    with pytest.raises((TypeError, ValueError)) as expected:
        originals[operation](value, n_points=8)
    with pytest.raises(type(expected.value)) as actual:
        natives[operation](value, n_points=8)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("real", [False, True])
def test_original_fft_consumer_with_native_helpers(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, real: bool
) -> None:
    pytestconfig.addinivalue_line(
        "markers", "notebook: original upstream documentation fixture"
    )
    from qualtran.bloqs.qsp import fft_qsp_test

    monkeypatch.setattr(fft_qsp_test, "random_qsp_polynomial", random_qsp_polynomial)
    monkeypatch.setattr(
        fft_qsp_test,
        "check_gqsp_polynomial_pair_on_random_points_on_unit_circle",
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    )
    consumer: Callable[[int, int], None] = (
        fft_qsp_test.test_real_polynomial_has_real_complementary_polynomial
        if real
        else fft_qsp_test.test_complementary_polynomial
    )
    consumer(3, 500)


def test_original_grid_consumer_with_default_native_fft(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    pytestconfig.addinivalue_line(
        "markers", "notebook: original upstream documentation fixture"
    )
    from qualtran.bloqs.qsp import generalized_qsp_test

    for name, native in (
        ("scale_down_to_qsp_polynomial", scale_down_to_qsp_polynomial),
        ("assert_is_qsp_polynomial", assert_is_qsp_polynomial),
        (
            "check_gqsp_polynomial_pair_on_random_points_on_unit_circle",
            check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
        ),
    ):
        monkeypatch.setattr(generalized_qsp_test, name, native)
    consumer: Callable[[float, float], None] = (
        generalized_qsp_test.test_complementary_polynomials_for_jacobi_anger_approximations
    )
    consumer(2, 1e-4)


@pytest.mark.parametrize("generator", [False, True])
@pytest.mark.parametrize("degree", [0, 4, -1])
def test_rng_implementation_and_failures_match_source(
    generator: bool, degree: int
) -> None:
    from qualtran.linalg.polynomial.qsp_testing import random_qsp_polynomial as source

    outcomes = []
    draws = []
    for evaluator in (source, random_qsp_polynomial):
        # The source dynamically accepts both NumPy RNG implementations.
        rng: Any = np.random.default_rng(91) if generator else np.random.RandomState(91)
        value: object
        try:
            value = evaluator(degree, random_state=rng)
        except ValueError as error:
            value = type(error), str(error)
        outcomes.append(value)
        draws.append(rng.random(9))
    np.testing.assert_array_equal(outcomes[0], outcomes[1])
    np.testing.assert_array_equal(draws[0], draws[1])


@pytest.mark.parametrize("operation", ["maximum", "scale", "assert"])
def test_shape_only_grid_failure_keeps_native_type_name(operation: str) -> None:
    from qualtran.linalg.polynomial import qsp_testing as source
    from qualtran.symbolics import Shaped

    originals: dict[str, Callable[..., object]] = {
        "maximum": source._polynomial_max_abs_value_on_unit_circle,
        "scale": source.scale_down_to_qsp_polynomial,
        "assert": source.assert_is_qsp_polynomial,
    }
    natives: dict[str, Callable[..., object]] = {
        "maximum": polynomial_max_abs_value_on_unit_circle,
        "scale": scale_down_to_qsp_polynomial,
        "assert": assert_is_qsp_polynomial,
    }
    with pytest.raises(TypeError) as expected:
        originals[operation](Shaped((2,)), n_points=8)
    with pytest.raises(TypeError) as actual:
        natives[operation](SymbolicArrayShape((2,)), n_points=8)
    assert str(actual.value).replace("SymbolicArrayShape", "Shaped") == str(
        expected.value
    )


@pytest.mark.parametrize("kind", ["subclass", "masked"])
def test_array_conversion_and_unscaled_identity_match_source(kind: str) -> None:
    from qualtran.linalg.polynomial.qsp_testing import (
        scale_down_to_qsp_polynomial as source,
    )

    class Array(np.ndarray[tuple[int, ...], np.dtype[np.float32]]):
        pass

    raw = np.array([0.125, 0.125], dtype=np.float32)
    masked: Callable[[NDArray[np.float32]], NDArray[Any]] = np.ma.array
    data = raw.view(Array) if kind == "subclass" else masked(raw)
    expected = source(data, n_points=8)
    actual = scale_down_to_qsp_polynomial(data, n_points=8)
    assert type(actual) is type(expected) is np.ndarray
    assert np.shares_memory(actual, data) == np.shares_memory(expected, data)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
