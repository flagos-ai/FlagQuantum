"""Shared scalar width mathematics preserves pinned source outcomes and types."""

import math
import warnings

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)
X = sympy.Symbol("x", nonnegative=True)
VALUES = [
    -2,
    -1,
    -0.25,
    0,
    0.25,
    False,
    True,
    1,
    3,
    2**53 - 1,
    2**53 + 1,
    2**64,
    2**10000,
    float("nan"),
    float("inf"),
    float("-inf"),
    np.int64(3),
    np.float32(0.25),
    np.float64(1.25),
    np.bool_(True),
    sympy.Integer(3),
    sympy.Integer(-1),
    sympy.Rational(1, 3),
    N,
    X,
]


def source_contract(name, value):
    from qualtran.symbolics import math_funcs

    return getattr(math_funcs, name)(value)


def native_math(name, value):
    from flagquantum.core import value_math

    return getattr(value_math, name)(value)


@pytest.fixture(params=["source_contract", "native"])
def calculate(request):
    return source_contract if request.param == "source_contract" else native_math


def outcome(calculate, name, value):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        try:
            result = calculate(name, value)
        except (TypeError, ValueError, OverflowError) as error:
            value_outcome = type(error), str(error)
        else:
            result_type = type(result)
            if isinstance(result, float) and math.isnan(result):
                result = "nan"
            value_outcome = result_type, result
    return value_outcome, tuple(
        (warning.category, str(warning.message)) for warning in emitted
    )


@pytest.mark.parametrize("name", ["ceil", "floor", "log2", "bit_length"])
@pytest.mark.parametrize("value", VALUES)
def test_scalar_values_types_diagnostics_and_warnings(calculate, name, value):
    assert outcome(calculate, name, value) == outcome(source_contract, name, value)


@pytest.mark.parametrize("name", ["ceil", "floor", "log2", "bit_length"])
@pytest.mark.parametrize("symbolic", [False, True])
def test_integer_subclass_protocol_evaluation_order(calculate, name, symbolic):
    events = []

    class Number(int):
        def is_symbolic(self):
            events.append("symbolic")
            return symbolic

    value = Number(3)
    expected = source_contract(name, value)
    expected_events = events.copy()
    events.clear()
    actual = calculate(name, value)
    assert type(actual) is type(expected)
    assert actual == expected
    assert events == expected_events


def test_bit_length_zero_interval_short_circuits_rounding(calculate):
    class Number(float):
        def __floor__(self):
            raise AssertionError("Zero interval must bypass floor")

    assert calculate("bit_length", Number(0.25)) == 0


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("ceil", -1.25, -1),
        ("floor", -1.25, -2),
        ("log2", 2**10000, 10000),
        ("bit_length", 1023, 10),
        ("bit_length", 0.25, 0),
    ],
)
def test_concrete_native_math_without_optional_sympy(
    name, value, expected, monkeypatch
):
    import builtins

    original_import = builtins.__import__

    def guarded_import(module, *args, **kwargs):
        if module == "sympy" or module.startswith("sympy."):
            raise AssertionError("Concrete built-ins must not import optional Sympy")
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert native_math(name, value) == expected
