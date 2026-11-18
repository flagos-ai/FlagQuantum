"""Complete pinned symbolic mathematics retains scalar and consumer semantics."""

import warnings
from collections.abc import Callable
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sympy

from flagquantum.core import value_math as native
from flagquantum.core.value_parameters import is_symbolic_value
from flagquantum.simulation.value_dimensions import (
    SymbolicArrayShape,
    value_length,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
NAMES = ("ln", "sexp", "sarg", "sabs", "ssqrt", "acos", "sconj", "is_zero")


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran import symbolics

    for name, blob in (
        ("math_funcs.py", "0da09c48fb44a547835bdf3eb40b44857133e10a"),
        ("__init__.py", "95cf224f0dfafe5db50d3713865ddba9635dd4c0"),
        ("types.py", "aaa928ee3070f26ae8f3e78ab399d536f5b5dc35"),
        ("types_test.py", "8e6656119aef065c6a1dfc7143ba4b139751190d"),
        ("math_funcs_test.py", "720fa16e632d2fb2f1b231a85821c27accb3971c"),
    ):
        raw = (Path(symbolics.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def outcome(function: Callable[..., Any], *values: object) -> tuple[object, ...]:
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        try:
            result = function(*values)
        except (TypeError, ValueError, OverflowError, AttributeError) as error:
            record = ("error", type(error), str(error))
        else:
            record = ("value", type(result), result)
    return (*record, tuple((w.category, str(w.message)) for w in emitted))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "value",
    (
        -2,
        -0.0,
        0,
        0.25,
        1,
        2,
        2**1000,
        True,
        1j,
        1 + 2j,
        float("nan"),
        float("inf"),
        np.float32(0.25),
        np.longdouble(0.25),
        np.complex64(1j),
        np.int64(3),
        np.array([0.25, 0.5]),
        sympy.Integer(2),
        sympy.Rational(1, 3),
        sympy.I,
        sympy.Symbol("x"),
        None,
    ),
)
def test_remaining_scalar_functions_preserve_types_values_and_diagnostics(
    name: str, value: object
) -> None:
    from qualtran import symbolics as source

    expected = outcome(getattr(source, name), value)
    actual = outcome(getattr(native, name), value)
    assert actual[:2] == expected[:2] and actual[3] == expected[3]
    if actual[0] == "error" or isinstance(actual[2], sympy.Basic):
        assert actual[2] == expected[2]
    else:
        np.testing.assert_equal(actual[2], expected[2])


def test_pi_short_circuits_and_zero_retains_unknown_verdict() -> None:
    from qualtran.symbolics import is_zero, pi

    calls: list[str] = []

    class Probe:
        def is_symbolic(self) -> bool:
            calls.append("symbolic")
            return True

        def equals(self, value: int) -> None:
            calls.append(f"equals({value})")
            return

    probe = Probe()
    for function in (pi, native.pi):
        calls.clear()
        assert function(probe, object()) is sympy.pi
        assert calls == ["symbolic"]
    for function in (is_zero, native.is_zero):
        calls.clear()
        assert function(probe) is None
        assert calls == ["symbolic", "equals(0)"]
    assert type(native.pi()) is type(pi())
    assert native.pi() == pi()

    class Failure(Probe):
        def equals(self, value: int) -> None:
            raise ValueError("undecidable")

    assert native.is_zero(Failure()) is is_zero(Failure()) is False

    class Broken(Probe):
        def equals(self, value: int) -> None:
            raise RuntimeError("broken symbolic evaluation")

    for function in (is_zero, native.is_zero):
        with pytest.raises(RuntimeError, match="broken symbolic evaluation"):
            function(Broken())


def test_all_original_math_tests_use_native_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.symbolics import math_funcs_test as original

    for name in (
        "ln",
        "log2",
        "sexp",
        "sarg",
        "ssqrt",
        "ceil",
        "smax",
        "smin",
        "bit_length",
        "is_zero",
    ):
        monkeypatch.setattr(original, name, getattr(native, name))
    for name in (
        "log2",
        "ln",
        "ssqrt",
        "sexp",
        "sarg",
        "ceil",
        "smax",
        "smin",
        "bit_length",
        "bit_length_symbolic_simplify",
        "is_zero",
    ):
        getattr(original, f"test_{name}")()
    for value in (3, 4):
        original.test_bit_length_symbolic(value)


def test_original_shape_tests_use_existing_native_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.symbolics import types_test as original

    monkeypatch.setattr(original, "Shaped", SymbolicArrayShape)
    monkeypatch.setattr(original, "slen", value_length)
    monkeypatch.setattr(original, "is_symbolic", is_symbolic_value)
    for shape in (
        (4,),
        (1, 2),
        (1, 2, 3),
        (sympy.Symbol("n"),),
        (sympy.Symbol("n"), sympy.Symbol("m"), 100),
    ):
        original.test_shaped(shape)


def test_actual_phase_and_su2_consumers_accept_native_math_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.basic_gates import global_phase, su2_rotation

    from tests.unit.test_global_phase_source_binding import (
        test_whole_scalar_source_directions_carrier_and_independent_model,
    )
    from tests.unit.test_su2_source_binding import (
        test_actual_su2_directions_source_stream_complex_columns_and_models,
    )

    for name in ("sexp", "sarg", "pi"):
        monkeypatch.setattr(global_phase, name, getattr(native, name))
    monkeypatch.setattr(su2_rotation, "pi", native.pi)
    test_whole_scalar_source_directions_carrier_and_independent_model(-0.7)
    test_actual_su2_directions_source_stream_complex_columns_and_models(
        (0.2, 0.3, -0.4, 7.2)
    )
