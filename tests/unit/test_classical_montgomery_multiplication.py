"""Unmodified Montgomery window table, integer action, history and reverse checks."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_montgomery_multiplication import (
    montgomery_multiply_classical_values,
    montgomery_window_classical_values,
)
from tests.unit.test_classical_modular_addition import compare_value_calls
from tests.unit.test_classical_product_accumulation import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = [
    (3, 1, 3),
    (4, 1, 5),
    (4, 2, 5),
    (4, 4, 5),
    (5, 1, 9),
    (5, 5, 9),
    (5, 1, 13),
    (5, 5, 13),
    (6, 2, 7),
    (6, 3, 7),
]


def original(width, window, modulus, uncompute=False):
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import (
        DirtyOutOfPlaceMontgomeryModMul,
    )

    return DirtyOutOfPlaceMontgomeryModMul(width, window, modulus, uncompute=uncompute)


@pytest.mark.parametrize("width,window,modulus", CASES)
def test_complete_small_value_domain_history_types_and_reverse_presence(
    width, window, modulus
):
    source = original(width, window, modulus)
    for left in range(-1, modulus + 2):
        for right in range(-1, modulus + 2):

            def expected(x, y):
                return tuple(source.on_classical_vals(x=x, y=y).values())

            def actual(x, y):
                return tuple(
                    montgomery_multiply_classical_values(
                        x, y, width, window, modulus
                    ).values()
                )

            compare_value_calls(expected, actual, (left, right))
    # The unchanged source may assert for composite modulus or overflowing
    # diagnostic values. Successful outputs retain exact scalar result types.
    try:
        result = source.on_classical_vals(x=1, y=1)
    except AssertionError:
        # The source Fermat-based assertion also fails on some composite
        # moduli. This is retained, not replaced by inverse-modulus math.
        with pytest.raises(AssertionError):
            montgomery_multiply_classical_values(1, 1, width, window, modulus)
        return
    native = montgomery_multiply_classical_values(1, 1, width, window, modulus)
    assert tuple(native) == tuple(result)
    erased = montgomery_multiply_classical_values(
        1,
        1,
        width,
        window,
        modulus,
        target=native["target"],
        history=native["qrom_indices"],
        reduced=native["reduced"],
        uncompute=True,
    )
    assert erased == {"x": 1, "y": 1}


@pytest.mark.parametrize("width,window,modulus", CASES)
@pytest.mark.parametrize(
    "target,history", [(0, 0), (7, 3), (-1, 2), (np.int64(5), np.int64(1))]
)
def test_source_integer_window_preserves_numpy_table_and_bit_order(
    width, window, modulus, target, history
):
    source = original(width, window, modulus)
    for left in range(1 << window):

        def expected(x, y, t, h):
            return source._classical_action_window(x, y, t, h)

        def actual(x, y, t, h):
            return montgomery_window_classical_values(
                x, y, t, h, width, window, modulus
            )

        compare_value_calls(expected, actual, (left, 2, target, history))


@pytest.mark.parametrize(
    "params",
    [
        (2, 1, 3),
        (4, 0, 5),
        (4, 3, 5),
        (4, 2, 4),
        (sympy.Integer(4), 2, 5),
        (4, sympy.Integer(2), 5),
        (4, 2, sympy.Integer(5)),
        (sympy.Symbol("n"), 1, 3),
        (4, 1, sympy.Symbol("p")),
    ],
)
def test_original_constructor_and_symbolic_action_rejections(params):
    width, window, modulus = params
    try:
        source = original(*params)
        source.on_classical_vals(x=1, y=1)
    except Exception as error:
        with pytest.raises(type(error)):
            montgomery_multiply_classical_values(1, 1, *params)
    else:
        raise AssertionError("This case must exercise a source failure")


@pytest.mark.parametrize("uncompute", [False, True])
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"target": 1},
        {"target": 1, "qrom_indices": 0},
        {"target": 1, "qrom_indices": 99, "reduced": 7},
        {"target": 2, "qrom_indices": 0, "reduced": 0},
    ],
)
def test_original_forward_auxiliary_assertions_and_weak_reverse_diagnostics(
    fields, uncompute
):
    source = original(4, 2, 5, uncompute)
    native_fields = {
        "history" if key == "qrom_indices" else key: value
        for key, value in fields.items()
    }
    try:
        expected = source.on_classical_vals(x=1, y=1, **fields)
    except Exception as error:
        with pytest.raises(type(error)):
            montgomery_multiply_classical_values(
                1, 1, 4, 2, 5, uncompute=uncompute, **native_fields
            )
    else:
        actual = montgomery_multiply_classical_values(
            1, 1, 4, 2, 5, uncompute=uncompute, **native_fields
        )
        assert tuple(actual) == tuple(expected)
        for key in expected:
            same_value(actual[key], expected[key])


def test_lazy_table_cache_and_replaceable_window_dependency(monkeypatch):
    import flagquantum.simulation.classical_montgomery_multiplication as module

    original_data = module._inversion_data
    seen = []

    def data(window, modulus):
        seen.append((window, modulus))
        result = original_data(window, modulus)
        np.testing.assert_array_equal(result, original(4, 2, 5)._inversion_data)
        return result

    monkeypatch.setattr(module, "_inversion_data", data)
    module.montgomery_multiply_classical_values(2, 3, 4, 2, 5)
    assert seen == [(2, 5)]  # Two windows share the same lazy array.
    seen.clear()
    module.montgomery_multiply_classical_values(0, 3, 4, 2, 5)
    assert seen == []  # Exterior branch never creates the table.
    monkeypatch.setattr(
        module,
        "_inversion_data",
        lambda window, modulus: np.zeros(1 << window, dtype=np.int64),
    )
    replaced = module.montgomery_window_classical_values(1, 1, 0, 0, 4, 2, 5)
    assert replaced != original(4, 2, 5)._classical_action_window(1, 1, 0, 0)


def test_numpy_table_overflow_is_not_replaced_by_python_integer_arithmetic():
    params = (66, 1, 2**63 + 1)
    source = original(*params)
    with pytest.raises((OverflowError, TypeError)) as error:
        source.on_classical_vals(x=1, y=1)
    with pytest.raises(type(error.value)):
        montgomery_multiply_classical_values(1, 1, *params)


@pytest.mark.parametrize(
    "left,right",
    [
        (np.int8(2), np.int8(3)),
        (np.uint8(2), np.uint8(3)),
        (np.array([1]), np.array([2])),
        (np.array([1, 2]), np.array([2, 3])),
        (0, np.uint8(1)),
    ],
)
def test_root_scalar_types_array_diagnostics_and_original_object_values(left, right):
    source = original(4, 2, 5)

    def expected(x, y):
        return tuple(source.on_classical_vals(x=x, y=y).values())

    def actual(x, y):
        return tuple(montgomery_multiply_classical_values(x, y, 4, 2, 5).values())

    compare_value_calls(expected, actual, (left, right))
