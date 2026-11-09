"""Shared arithmetic follows the pinned source, including protocol edge cases."""

import builtins
import math

import numpy as np
import pytest
import sympy

from flagquantum.compiler._component_trace import symbolic_math as compiler_math
from flagquantum.core import value_math

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def compare_result(source, native, arguments):
    results = []
    for function in (source, native):
        try:
            value = function(*arguments())
            results.append(("value", type(value), value))
        except Exception as error:
            results.append(("error", type(error), str(error)))
    expected, actual = results
    assert actual[:2] == expected[:2]
    if actual[0] == "value" and isinstance(actual[2], float) and math.isnan(actual[2]):
        assert math.isnan(expected[2])
    else:
        assert actual[2] == expected[2]


class SymbolicInteger(int):
    def is_symbolic(self):
        return True


class ConcreteInteger(int):
    def is_symbolic(self):
        return False


@pytest.mark.parametrize("name", ["smax", "smin"])
@pytest.mark.parametrize("iterable", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        (),
        (3,),
        (3, 2, -1),
        (True, False),
        (3.0, 2),
        (float("nan"), 3.0),
        (3.0, float("nan")),
        (np.int64(3), np.int64(2)),
        (np.float32(3), np.float32(2)),
        (sympy.Integer(3), 2),
        (sympy.Symbol("n", positive=True), 2),
        (SymbolicInteger(3), 2),
        (ConcreteInteger(3), 2),
        ("a", "b"),
        (1j, 2j),
    ],
)
def test_extrema_results_errors_types_and_iterator_consumption(name, iterable, values):
    from qualtran import symbolics

    def arguments():
        return (iter(values),) if iterable else values

    compare_result(getattr(symbolics, name), getattr(value_math, name), arguments)
    assert getattr(compiler_math, name) is getattr(value_math, name)


@pytest.mark.parametrize("name", ["smax", "smin"])
def test_singleton_identity_and_full_iterator_consumption(name):
    from qualtran import symbolics

    value = object()
    assert getattr(value_math, name)(iter((value,))) is value
    for function in (getattr(symbolics, name), getattr(value_math, name)):
        seen = []

        def items(seen):
            for item in (3, 2, 5):
                seen.append(item)
                yield item

        assert function(items(seen)) == (5 if name == "smax" else 2)
        assert seen == [3, 2, 5]


@pytest.mark.parametrize("name", ["ssum", "prod"])
@pytest.mark.parametrize(
    "values", [(), (1, 2, 3), (np.int64(2), np.int64(3)), (sympy.Symbol("n"), 2)]
)
def test_ordered_folds_retain_integer_identity_and_scalar_type(name, values):
    from qualtran import symbolics

    compare_result(
        getattr(symbolics, name), getattr(value_math, name), lambda: (iter(values),)
    )


@pytest.mark.parametrize("name", ["ssum", "prod"])
def test_folds_use_source_left_association_and_operator_order(name):
    from qualtran import symbolics

    traces = []
    for function in (getattr(symbolics, name), getattr(value_math, name)):
        trace = []

        class Number:
            def __init__(self, value, log):
                self.value = value
                self.log = log

            def __add__(self, other):
                self.log.append(("add", self.value, other.value))
                return Number(self.value + other.value, self.log)

            def __radd__(self, other):
                self.log.append(("radd", other, self.value))
                return Number(other + self.value, self.log)

            def __mul__(self, other):
                self.log.append(("mul", self.value, other.value))
                return Number(self.value * other.value, self.log)

            def __rmul__(self, other):
                self.log.append(("rmul", other, self.value))
                return Number(other * self.value, self.log)

        result = function(Number(value, trace) for value in (2, 3, 5))
        traces.append((result.value, trace))
    assert traces[0] == traces[1]


def test_plain_native_width_math_does_not_import_optional_sympy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sympy":
            raise AssertionError("Unexpected optional import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert value_math.smax(3, 2) == compiler_math.smax(3, 2) == 3
    assert value_math.smin((3, 2)) == 2
    assert value_math.ssum((2, 3)) == 5
    assert value_math.prod((2, 3)) == 6
