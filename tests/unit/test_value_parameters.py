"""Source symbolic protocol order, raw checker results and real codec consumers."""

import builtins

import numpy as np
import pytest
import sympy

from flagquantum.simulation.value_parameters import is_symbolic_value
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class SymbolicInteger(int):
    def is_symbolic(self):
        return True


class ConcreteInteger(int):
    def is_symbolic(self):
        return False


class Probe:
    def __init__(self, name, result, trace):
        self.name, self.result, self.trace = name, result, trace

    def is_symbolic(self):
        self.trace.append(self.name)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        1,
        True,
        1.5,
        1j,
        "x",
        [1],
        (1,),
        {"x": 1},
        np.int64(3),
        np.bool_(True),
        np.array([1, 2]),
        sympy.Integer(3),
        sympy.Symbol("n"),
        sympy.S.true,
        SymbolicInteger(3),
        ConcreteInteger(3),
    ],
)
def test_source_single_parameter_result_type_and_value(value):
    from qualtran.symbolics import is_symbolic

    compare_call(is_symbolic, is_symbolic_value, value)


@pytest.mark.parametrize(
    "result", [True, False, None, 0, 1, np.bool_(True), [], [1], sympy.Symbol("n") > 0]
)
def test_single_checker_result_is_not_coerced_or_copied(result):
    from qualtran.symbolics import is_symbolic

    trace = []
    probe = Probe("value", result, trace)
    expected = is_symbolic(probe)
    actual = is_symbolic_value(probe)
    assert actual is expected is result
    assert trace == ["value", "value"]


@pytest.mark.parametrize(
    "results",
    [
        (),
        (False, False),
        (False, True, RuntimeError("unused")),
        (True, RuntimeError("unused")),
        (None, np.bool_(True)),
        (False, RuntimeError("called")),
        (False, sympy.Symbol("n") > 0),
    ],
)
def test_variadic_any_preserves_callback_order_errors_and_boolean_result(results):
    from qualtran.symbolics import is_symbolic

    traces = [[], []]
    outcomes = []
    for classifier, trace in zip((is_symbolic, is_symbolic_value), traces, strict=True):
        values = [Probe(str(i), result, trace) for i, result in enumerate(results)]
        try:
            outcomes.append(("result", classifier(*values)))
        except Exception as error:
            outcomes.append(("error", type(error), str(error)))
    assert outcomes[0] == outcomes[1]
    assert traces[0] == traces[1]
    if outcomes[0][0] == "result":
        assert type(outcomes[0][1]) is type(outcomes[1][1]) is bool


@pytest.mark.parametrize("checker", [None, False, 1, "not_callable"])
def test_source_absent_or_noncallable_checker_attribute(checker):
    from qualtran.symbolics import is_symbolic

    class Value:
        is_symbolic = checker

    compare_call(is_symbolic, is_symbolic_value, Value())


def test_sympy_basic_precedes_checker_lookup():
    from qualtran.symbolics import is_symbolic

    class Basic(sympy.Basic):
        @property
        def is_symbolic(self):
            pytest.fail("A Sympy Basic must be recognized before checker lookup")

    assert is_symbolic(Basic()) is is_symbolic_value(Basic()) is True


def test_integer_subclass_symbolic_protocol_reaches_all_numerical_consumers():
    from qualtran import QFxp, QInt, QMontgomeryUInt
    from qualtran._infra.data_types import _BUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul

    from flagquantum.simulation.bounded_integer_values import (
        validate_bounded_integer_shape,
    )
    from flagquantum.simulation.classical_montgomery_window import (
        montgomery_window_bits_classical_values,
    )
    from flagquantum.simulation.fixed_point_encoding import fixed_point_integer_width
    from flagquantum.simulation.integer_encoding import integer_to_bits
    from flagquantum.simulation.montgomery_values import (
        integer_to_montgomery_value,
        montgomery_value_domain,
    )

    width = SymbolicInteger(3)
    compare_call(
        lambda x: QInt(width).to_bits(x),
        lambda x: integer_to_bits(x, width, signed=True),
        1,
    )
    compare_call(
        lambda _: QFxp(width, width, signed=True).num_int,
        lambda _: fixed_point_integer_width(width, width, signed=True),
        None,
    )
    compare_call(
        lambda _: _BUInt(width, 9) and None,
        lambda _: validate_bounded_integer_shape(width, 9),
        None,
    )
    modulus = SymbolicInteger(5)
    compare_call(
        lambda x: QMontgomeryUInt(3, modulus).uint_to_montgomery(x),
        lambda x: integer_to_montgomery_value(x, 3, modulus),
        1,
    )
    compare_call(
        lambda _: QMontgomeryUInt(3, modulus)._bit_encoding.get_domain(),
        lambda _: montgomery_value_domain(3, modulus),
        None,
    )
    compare_call(
        lambda _: SingleWindowModMul(1, width, 5).on_classical_vals(
            x=[0], y=1, target=[0] * 4, qrom_index=0
        ),
        lambda _: montgomery_window_bits_classical_values(
            [0], 1, [0] * 4, 0, width, 1, 5
        ),
        None,
    )


def test_concrete_native_codec_does_not_require_optional_sympy(monkeypatch):
    from flagquantum.simulation.integer_encoding import (
        integer_array_to_bits,
        integer_to_bits,
    )

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sympy":
            raise ModuleNotFoundError("No module named 'sympy'", name="sympy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert integer_to_bits(-1, 3, signed=True) == [1, 1, 1]
    np.testing.assert_array_equal(
        integer_array_to_bits(np.array([1]), np.int64(3)), [[0, 0, 1]]
    )
    assert is_symbolic_value(SymbolicInteger(3)) is True


def test_broken_installed_sympy_dependency_is_not_hidden(monkeypatch):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "sympy":
            raise ModuleNotFoundError(
                "Missing nested dependency", name="nested_dependency"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ModuleNotFoundError) as error:
        is_symbolic_value(3)
    assert error.value.name == "nested_dependency"
