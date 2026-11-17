"""Differential coverage for the locked rotation synthesis numerical owners."""

from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flagquantum.synth._rotation_math import MathConfig, NumpyConfig, isclose, with_dps
from flagquantum.synth._rotation_types import Real, is_int

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def source_modules() -> tuple[Any, Any]:
    math = pytest.importorskip("qualtran.rotation_synthesis._math_config")
    types = pytest.importorskip("qualtran.rotation_synthesis._typing")
    for module, expected in (
        (math, "ea435fa802e0dabc272d3ac52065b709fd441b55"),
        (types, "66bdc937ad691a24bb523d64c2d77cc34de36031"),
    ):
        content = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == expected
    return math, types


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("backend", ["vanilla", "numpy", 20, 60])
def test_every_numeric_operation_uses_the_locked_backend(
    backend: str | int, source_modules: tuple[Any, Any]
) -> None:
    import mpmath

    math, _ = source_modules
    with mpmath.workdps(mpmath.mp.dps):
        if backend == "vanilla":
            native, source = MathConfig(), math.MathConfig()
        elif backend == "numpy":
            native, source = NumpyConfig, math.NumpyConfig
        else:
            assert isinstance(backend, int)
            native, source = with_dps(backend), math.with_dps(backend)
        assert native.name == source.name
        assert hash(native) == hash(source)
        for name in ("zero", "one", "pi", "sqrt2"):
            actual, expected = getattr(native, name), getattr(source, name)
            assert type(actual) is type(expected)
            assert actual == expected
        for value in ("0.125", "0.79", "1.0000000000000001", "-0.75", "2"):
            x, y = native.number(value), source.number(value)
            assert type(x) is type(y)
            assert x == y
            operations = [
                ("floor", (x,), (y,)),
                ("ceil", (x,), (y,)),
                ("cos", (x,), (y,)),
                ("sin", (x,), (y,)),
                ("arctan2", (x, -native.one), (y, -source.one)),
                (
                    "isclose",
                    (x, x + native.number("1e-12")),
                    (y, y + source.number("1e-12")),
                ),
            ]
            if x > 0:
                operations.extend([("log", (x,), (y,)), ("sqrt", (x,), (y,))])
            if abs(x) < 1:
                operations.extend([("arcsin", (x,), (y,)), ("arccos", (x,), (y,))])
            for op, actual_args, expected_args in operations:
                actual = getattr(native, op)(*actual_args)
                expected = getattr(source, op)(*expected_args)
                assert type(actual) is type(expected)
                assert actual == expected


@pytest.mark.ftqc_reference
def test_precision_change_is_observed_by_an_existing_configuration(
    source_modules: tuple[Any, Any],
) -> None:
    import mpmath

    math, _ = source_modules
    results = []
    for configure in (with_dps, math.with_dps):
        with mpmath.workdps(15):
            early = configure(20)
            first = early.sqrt(early.number(2))
            assert early.isclose(early.zero, early.number("1e-10"))
            second = configure(60)
            assert mpmath.mp.dps == 60
            assert early.name == "mpmath_20" and second.name == "mpmath_60"
            assert not early.isclose(early.zero, early.number("1e-10"))
            late = early.sqrt(early.number(2))
            assert late != first
            results.append((first, late, early.sqrt2, second.sqrt2))
    assert results[0] == results[1]


@pytest.mark.ftqc_reference
def test_tolerance_and_integer_predicates_match_source(
    source_modules: tuple[Any, Any],
) -> None:
    import mpmath

    math, types = source_modules
    for x, y, eps in [(8.0, 9.0, 0.125), (9.0, 8.0, 0.125), (0.0, 0.0, 0.0)]:
        assert isclose(x, y, eps) == math.isclose(x, y, eps)
    assert not isclose(8.0, 9.0, 0.125)
    assert isclose(9.0, 8.0, 0.125)
    values = (
        True,
        np.bool_(True),
        1,
        np.int64(-2),
        np.uint64(3),
        1.0,
        np.float64(1),
        mpmath.mpf(1),
        mpmath.mpc(1),
    )
    assert [is_int(value) for value in values] == [
        types.is_int(value) for value in values
    ]


def test_configuration_keeps_callable_dispatch_and_name_hash() -> None:
    left = MathConfig("custom", sqrt=lambda value: value * 2)
    right = MathConfig("custom")
    assert left.sqrt(3) == 6
    assert left != right
    assert hash(left) == hash(right) == hash("custom")
    assert type(NumpyConfig.number("1.0000000000000001")) is np.longdouble


@pytest.mark.ftqc_reference
def test_custom_backend_constructor_retains_source_keyword_names(
    source_modules: tuple[Any, Any],
) -> None:
    source_math, _ = source_modules

    def double(value: Real) -> Real:
        return value * 2

    native = MathConfig(name="custom", sqrt=double, arccos=double)
    source = source_math.MathConfig(name="custom", sqrt=double, arccos=double)
    assert native.sqrt(x=3) == source.sqrt(x=3)
    assert native.arccos(x=4) == source.arccos(x=4)
    assert hash(native) == hash(source)
